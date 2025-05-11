import logging
import math
import os.path
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage.transform import rotate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from ace_network import Regressor
from ace_util import get_coord, to_tensor

_logger = logging.getLogger(__name__)


class CamLocDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(
            self,
            root_dir,
            mode=0,
            sparse=False,
            augment=False,
            aug_rotation=15,
            aug_scale_min=2 / 3,
            aug_scale_max=3 / 2,
            aug_black_white=0.1,
            aug_color=0.3,
            image_height=480,
            use_half=True,
    ):
        """Constructor.

        Parameters:
            root_dir: Folder of the data (training or test).
            mode:
                0 = RGB only, load no initialization targets. Default for the ACE paper.
                1 = RGB + ground truth scene coordinates, load or generate ground truth scene coordinate targets
                2 = RGB-D, load camera coordinates instead of scene coordinates
            sparse: for mode = 1 (RGB+GT SC), load sparse initialization targets when True, load dense depth maps and
                generate initialization targets when False
            augment: Use random data augmentation, note: not supported for mode = 2 (RGB-D) since pre-generated eye
                coordinates cannot be augmented
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees.
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_min: Upper limit of image scale factor for uniform sampling
            aug_black_white: Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]
            aug_color: Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]
            image_height: RGB images are rescaled to this maximum height (if augmentation is disabled, and in the range
                [aug_scale_min * image_height, aug_scale_max * image_height] otherwise).
            use_half: Enabled if training with half-precision floats.
            num_clusters: split the input frames into disjoint clusters using hierarchical clustering in order to train
                an ensemble model. Clustering is deterministic, so multiple training calls with the same number of
                target clusters will result in the same split. See the paper for details of the approach. Disabled by
                default.
            cluster_idx: If num_clusters is not None, then use this parameter to choose the cluster used for training.
        """

        if '7scenes' in str(root_dir):
            self.scene = '7scenes'
        elif 'Cambridge' in str(root_dir):
            self.scene = 'Cambridge'

        self.use_half = use_half

        self.init = (mode == 1)
        self.sparse = sparse
        self.eye = (mode == 2)

        self.image_height = image_height

        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        if self.eye and self.augment and (self.aug_rotation > 0 or self.aug_scale_min != 1 or self.aug_scale_max != 1):
            # pre-generated eye coordinates cannot be augmented
            _logger.warning("WARNING: Check your augmentation settings. Camera coordinates will not be augmented.")

        # Setup data paths.
        root_dir = Path(root_dir)

        # Main folders.
        rgb_dir = root_dir / 'rgb'
        pose_dir = root_dir / 'poses'
        calibration_dir = root_dir / 'calibration'
        depth_dir = root_dir / 'depth'

        # Find all images. The assumption is that it only contains image files.
        # PosixPath('datasets/7scenes_chess/train/rgb/seq-01-frame-000000.color.png')
        self.rgb_files = sorted(rgb_dir.iterdir())

        # Find all ground truth pose files. One per image.
        self.pose_files = sorted(pose_dir.iterdir())

        # Load camera calibrations. One focal length per image.
        self.calibration_files = sorted(calibration_dir.iterdir())

        # Load depth_map
        if os.path.exists(depth_dir):
            self.depth_files = sorted(depth_dir.iterdir())
        else:
            self.depth_files = None

        if len(self.rgb_files) != len(self.pose_files):
            raise RuntimeError('RGB file count does not match pose file count!')

        if len(self.rgb_files) != len(self.calibration_files):
            raise RuntimeError('RGB file count does not match calibration file count!')

        # Create grid of 2D pixel positions used when generating scene coordinates from depth.
        if self.init and not self.sparse:
            self.prediction_grid = self._create_prediction_grid()
        else:
            self.prediction_grid = None

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        if self.augment:
            self.image_transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(int(self.image_height * scale_factor)),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=self.aug_black_white, contrast=self.aug_black_white),
                # saturation=self.aug_color, hue=self.aug_color),  # Disable colour augmentation.
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25]
                ),
            ])
        else:
            self.image_transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(self.image_height),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4],  # statistics calculated over 7scenes training set, should generalize fairly well
                    std=[0.25]
                ),
            ])

        # We use this to iterate over all frames. If clustering is enabled this is used to filter them.
        self.valid_file_indices = np.arange(len(self.rgb_files))

        # Calculate mean camera center (using the valid frames only).
        self.mean_cam_center = self._compute_mean_camera_center()

    @staticmethod
    def _create_prediction_grid():
        # Assumes all input images have a resolution smaller than 5000x5000.
        prediction_grid = np.zeros((2,
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE),
                                    math.ceil(5000 / Regressor.OUTPUT_SUBSAMPLE)))

        for x in range(0, prediction_grid.shape[2]):
            for y in range(0, prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * Regressor.OUTPUT_SUBSAMPLE
                prediction_grid[1, y, x] = y * Regressor.OUTPUT_SUBSAMPLE

        return prediction_grid

    @staticmethod
    def _resize_image(image, image_height):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height)
        return image

    @staticmethod
    def _resize_depth(depth, image_height):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        depth = cv2.resize(depth, image_height)
        return depth

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _cluster(self, num_clusters):
        """
        Clusters the dataset using hierarchical kMeans.
        Initialization:
            Put all images in one cluster.
        Interate:
            Pick largest cluster.
            Split with kMeans and k=2.
            Input for kMeans is the 3D median scene coordiante per image.
        Terminate:
            When number of target clusters has been reached.
        Returns:
            cam_centers: For each cluster the mean (not median) scene coordinate
            labels: For each image the cluster ID
        """
        num_images = len(self.pose_files)
        _logger.info(f'Clustering a dataset with {num_images} frames into {num_clusters} clusters.')

        # A tensor holding all camera centers used for clustering.
        cam_centers = np.zeros((num_images, 3), dtype=np.float32)
        for i in range(num_images):
            pose = self._load_pose(i)
            cam_centers[i] = pose[:3, 3]

        # Setup kMEans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS

        # Label of next cluster.
        label_counter = 0

        # Initialise list of clusters with all images.
        clusters = []
        clusters.append((cam_centers, label_counter, np.zeros(3)))

        # All images belong to cluster 0.
        labels = np.zeros(num_images)

        # iterate kMeans with k=2
        while len(clusters) < num_clusters:
            # Select largest cluster (list is sorted).
            cur_cluster = clusters.pop(0)
            label_counter += 1

            # Split cluster.
            cur_error, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

            # Update cluster list.
            cur_mask = (cur_labels == 0)[:, 0]
            cur_cam_centers0 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

            cur_mask = (cur_labels == 1)[:, 0]
            cur_cam_centers1 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

            cluster_labels = labels[labels == cur_cluster[1]]
            cluster_labels[cur_mask] = label_counter
            labels[labels == cur_cluster[1]] = cluster_labels

            # Sort updated list.
            clusters = sorted(clusters, key=lambda cluster: cluster[0].shape[0], reverse=True)

        # clusters are sorted but cluster indices are random, remap cluster indices to sorted indices
        remapped_labels = np.zeros(num_images)
        remapped_clusters = []

        for cluster_idx_new, cluster in enumerate(clusters):
            cluster_idx_old = cluster[1]
            remapped_labels[labels == cluster_idx_old] = cluster_idx_new
            remapped_clusters.append((cluster[0], cluster_idx_new, cluster[2]))

        labels = remapped_labels
        clusters = remapped_clusters

        cluster_centers = np.zeros((num_clusters, 3))
        cluster_sizes = np.zeros((num_clusters, 1))

        for cluster in clusters:
            # Compute distance of each cam to the center of the cluster.
            cam_num = cluster[0].shape[0]
            cam_data = np.zeros((cam_num, 3))
            cam_count = 0

            # First compute the center of the cluster (mean).
            for i, cam_center in enumerate(cam_centers):
                if labels[i] == cluster[1]:
                    cam_data[cam_count] = cam_center
                    cam_count += 1

            cluster_centers[cluster[1]] = cam_data.mean(0)

            # Compute the distance of each cam from the cluster center. Then average and square.
            cam_dists = np.broadcast_to(cluster_centers[cluster[1]][np.newaxis, :], (cam_num, 3))
            cam_dists = cam_data - cam_dists
            cam_dists = np.linalg.norm(cam_dists, axis=1)
            cam_dists = cam_dists ** 2

            cluster_sizes[cluster[1]] = cam_dists.mean()

            _logger.info("Cluster %i: %.1fm, %.1fm, %.1fm, images: %i, mean squared dist: %f" % (
                cluster[1], cluster_centers[cluster[1]][0], cluster_centers[cluster[1]][1],
                cluster_centers[cluster[1]][2],
                cluster[0].shape[0], cluster_sizes[cluster[1]]))

        _logger.info('Clustering done.')

        return cluster_centers, cluster_sizes, labels

    def _compute_mean_camera_center(self):
        mean_cam_center = torch.zeros((3,))

        for idx in self.valid_file_indices:
            pose = self._load_pose(idx)

            # Get the translation component.
            mean_cam_center += pose[0:3, 3]

        # Avg.
        mean_cam_center /= len(self)
        return mean_cam_center

    def _load_image(self, idx):
        # image = io.imread(self.rgb_files[idx])

        img = cv2.imread(str(self.rgb_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.scene == 'Cambridge':
            img = cv2.resize(img, (848, 480))

        return img

    def _load_pose(self, idx):
        # Stored as a 4x4 matrix.
        pose = np.loadtxt(self.pose_files[idx])
        pose = torch.from_numpy(pose).float()

        return pose

    def _load_depth(self, idx):
        depth = cv2.imread(str(self.depth_files[idx]), -1)
        if self.scene == 'Cambridge':
            depth = cv2.resize(depth, (848, 480))

        return depth.astype(np.float32)

    def _get_single_item(self, idx, image_height):
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image = self._load_image(idx)

        # Load depth
        if self.depth_files:
            depth = self._load_depth(idx)
        else:
            depth = np.zeros_like(image)

        # Load intrinsics.
        k = np.loadtxt(self.calibration_files[idx])
        if k.size == 1:
            focal_length = float(k)
            centre_point = None
        elif k.shape == (3, 3):
            k = k.tolist()
            focal_length = [k[0][0], k[1][1]]
            centre_point = [k[0][2], k[1][2]]
        else:
            raise Exception("Calibration file must contain either a 3x3 camera \
                intrinsics matrix or a single float giving the focal length \
                of the camera.")

        # The image will be scaled to image_height, adjust focal length as well.
        f_scale_factor = image_height / image.shape[0]
        if centre_point:
            centre_point = [c * f_scale_factor for c in centre_point]
            focal_length = [f * f_scale_factor for f in focal_length]
        else:
            focal_length *= f_scale_factor

        # Rescale image.
        image = self._resize_image(image, image_height)
        depth = self._resize_image(depth, image_height)
        depth_transform = transforms.ToTensor()
        depth = depth_transform(depth)

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image.size[1], image.size[0]))

        # Apply remaining transforms.
        image = self.image_transform(image)
        depth = depth.clone().detach()

        # Load pose.
        pose = self._load_pose(idx)

        # Apply data augmentation if necessary.
        if self.augment:
            # Generate a random rotation angle.
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            # Rotate input image and mask.
            image = self._rotate_image(image, angle, 1, 'reflect')
            depth = self._rotate_image(depth, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, 1, 'constant')

            # Rotate ground truth camera pose as well.
            angle = angle * math.pi / 180.
            # Create a rotation matrix.
            pose_rot = torch.eye(4)
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)

            # Apply rotation matrix to the ground truth camera pose.
            pose = torch.matmul(pose, pose_rot)

        # Convert to half if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # Invert the pose.
        pose_inv = pose.inverse()

        # Create the intrinsics matrix.
        intrinsics = torch.eye(3)

        # Hardcode the principal point to the centre of the image unless otherwise specified.
        if centre_point:
            intrinsics[0, 0] = focal_length[0]
            intrinsics[1, 1] = focal_length[1]
            intrinsics[0, 2] = centre_point[0]
            intrinsics[1, 2] = centre_point[1]
        else:
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = image.shape[2] / 2
            intrinsics[1, 2] = image.shape[1] / 2

        # Also need the inverse.
        intrinsics_inv = intrinsics.inverse()

        # calculate coord
        depth = depth.detach().numpy()[0]
        pose[0:3, 3] = pose[0:3, 3] * 1000.
        coord, mask = get_coord(depth, pose, intrinsics_inv)
        coord, mask = to_tensor(coord, mask)
        pose[0:3, 3] = pose[0:3, 3] / 1000.

        image_mask = image_mask * mask

        return image, image_mask, coord, pose, pose_inv, intrinsics, intrinsics_inv, str(self.rgb_files[idx])

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
            # scale_factor = 1 / scale_factor #inverse scale sampling, not used for ACE mapping
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors because we need
        # to apply the same scale factor to all of them.
        image_height = int(self.image_height * scale_factor)

        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i, image_height) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_height)
