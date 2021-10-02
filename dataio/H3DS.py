import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

try:
    from h3ds.dataset import H3DS
except ModuleNotFoundError:
    pass

from utils.io_util import load_rgb, load_mask


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 train_cameras: bool,
                 data_dir: str,
                 downscale: float = 1.,  # [H, W]
                 scale_radius: float = -1):
        super().__init__()

        scene_root = data_dir
        labels, intrinsics_all, poses = self.load_data(scene_root, downscale, scale_radius)
        image_paths = self.get_paths(os.path.join(scene_root, 'image'), labels)
        mask_paths = self.get_paths(os.path.join(scene_root, 'rigid_masks'), labels)

        self.train_cameras = train_cameras
        self.intrinsics_all = intrinsics_all
        self.c2w_all = poses

        self.n_images = len(image_paths)

        tmp_rgb = load_rgb(image_paths[0], downscale)
        _, self.H, self.W = tmp_rgb.shape

        self.rgb_images = []
        for path in tqdm(image_paths, desc='loading images...'):
            rgb = load_rgb(path, downscale)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for path in tqdm(mask_paths, desc='loading masks...'):
            object_mask = load_mask(path, downscale)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        ground_truth = {"rgb": self.rgb_images[idx]}  # (n, 3)
        sample = {
            "intrinsics": self.intrinsics_all[idx],
            "object_mask": self.object_masks[idx],
        }

        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)
        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
        return tuple(all_parsed)

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'jpg'):
        if 'image' in Path(root).name:
            names = [f"img_{name}" for name in names]
        elif 'mask' in Path(root).name:
            names = [f"mask_{name}" for name in names]
        else:
            raise Exception("Invalid dir " + str(root))
        paths = [os.path.join(root, f"{name}.{ext}") for name in names]
        for path in paths:
            assert os.path.exists(path), f"{path}"
        return paths

    def get_extrinsics(self) -> (List[str], torch.Tensor, torch.Tensor, torch.Tensor):
        R_cols, Ts = self.RT[:, :3, :3], self.RT[:, :3, -1]
        cam_poses = -np.einsum('bij,bi->bj', R_cols, Ts)

        return self.labels, torch.tensor(R_cols, dtype=torch.float32), torch.tensor(Ts, dtype=torch.float32), \
               torch.tensor(cam_poses, dtype=torch.float32)

    def load_data(self, scene_path, downscale=1.0, scale_radius=-1.0, views_config_id='32'):
        scene_path = Path(scene_path)
        h3ds_dir = scene_path.parent
        scene_id = scene_path.name
        h3ds = H3DS(path=h3ds_dir)
        views_idx = h3ds.helper._config['scenes'][scene_id]['default_views_configs'][views_config_id]
        labels = ['{0:04}'.format(idx) for idx in views_idx]
        _, images, masks, cameras = h3ds.load_scene(scene_id=scene_id, views_config_id=views_config_id)

        # K = np.array([camera[0] for camera in cameras]).mean(axis=0)
        # RT = np.array([np.linalg.inv(camera[1]) for camera in cameras])
        # RT[:, :2, :] *= -1

        intrinsics_all = []
        poses = []
        cam_center_norms = []
        for camera in cameras:
            intrinsics = torch.from_numpy(camera[0]).float()
            intrinsics[0, 2] /= downscale
            intrinsics[1, 2] /= downscale
            intrinsics[0, 0] /= downscale
            intrinsics[1, 1] /= downscale
            intrinsics_all.append(intrinsics)

            pose = torch.from_numpy(camera[1]).float()
            cam_center_norms.append(torch.linalg.norm(pose[:3, 3]))
            poses.append(pose)

        if scale_radius > 0:
            max_cam_norm = max(cam_center_norms)
            for i in range(len(poses)):
                poses[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)

        return labels, intrinsics_all, poses

    def get_gt_pose(self):
        return torch.stack(self.c2w_all, dim=0)


if __name__ == '__main__':
    def test():
        dataset = SceneDataset(
            False,
            './data/H3DS/5ae021f2805c0854',
            scale_radius=3.0)
        c2w = dataset.get_gt_pose().data.cpu().numpy()
        extrinsics = np.linalg.inv(c2w)
        camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()
        from tools.vis_camera import visualize
        visualize(camera_matrix, extrinsics)


    test()
