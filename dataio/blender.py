from pathlib import Path
import os
import json

import torch
import numpy as np
from tqdm import tqdm

from utils.io_util import load_mask, load_rgb, glob_imgs
from utils.rend_util import rot_to_quat, load_K_Rt_from_P


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 scale_radius=-1,
                 split='train'):

        assert split in ('train', 'val', 'test')
        data_dir = Path(data_dir)

        self.train_cameras = train_cameras

        # Read json with camera poses and image paths
        with open(data_dir / f"transforms_{split}.json", 'r') as f:
            metadata = json.load(f)

        # Load intrinsics
        self.H = round(metadata.get('image_height', 800) // downscale)
        self.W = round(metadata.get('image_width', 800) // downscale)
        camera_angle_x = float(metadata['camera_angle_x'])
        camera_angle_y = float(metadata.get('camera_angle_y', camera_angle_x))
        focal_x = float(.5 * self.W / np.tan(.5 * camera_angle_x))
        focal_y = float(.5 * self.H / np.tan(.5 * camera_angle_y))

        # Load extrinsics
        camera_poses = [frame['transform_matrix'] for frame in metadata['frames']]
        camera_poses = torch.tensor(camera_poses, dtype=torch.float32)

        # Load images
        image_paths = [data_dir / (frame['file_path'] + '.png') for frame in metadata['frames']]

        self.rgb_images = []
        self.object_masks = []

        for path in tqdm(image_paths, desc='loading images...'):
            image_with_mask = load_rgb(path, downscale)
            assert image_with_mask.shape[1:] == (self.H, self.W), \
                f"expected {path} to be {self.H} x {self.W}, but it's {image_with_mask.shape[1:]}"

            rgb = image_with_mask[:3].reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

            if image_with_mask.shape[0] == 4:
                object_mask = image_with_mask[3].reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

        # Convert everything to this repository's format
        self.intrinsics = torch.tensor([
            [focal_x, 0,       self.W / 2, 0],
            [0,       focal_y, self.H / 2, 0],
            [0,       0,       1,          0],
            [0,       0,       0,          1],
        ])
        self.c2w_all = camera_poses

        # Apply VolSDF's camera normalization (section B.1 in the paper)
        if scale_radius > 0:
            max_cam_norm = np.linalg.norm(self.c2w_all[:, :3, 3], axis=-1).max()
            self.c2w_all[:, :3, 3] *= (scale_radius / max_cam_norm / 1.1)

        # Invert Y and Z axis (for some reason, this repository needs this...)
        self.c2w_all[:, :3, 1] *= -1
        self.c2w_all[:, :3, 2] *= -1

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        sample = {
            "intrinsics": self.intrinsics,
        }
        if self.object_masks:
            sample["object_mask"] = self.object_masks[idx]
        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        return idx, sample, ground_truth


if __name__ == "__main__":
    dataset = SceneDataset(False, './data/nerf_synthetic/lego', scale_radius=3.0)
    c2w = dataset.c2w_all
    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = dataset.intrinsics # next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()
    from tools.vis_camera import visualize
    visualize(camera_matrix, extrinsics)
