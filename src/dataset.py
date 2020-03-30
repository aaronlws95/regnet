import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2

import src.calib as calib
import src.utils as utils

class Kitti_Dataset(Dataset):

    def __init__(self, params):

        base_path = params['base_path']
        date = params['date']
        drives = params['drives']

        self.h_fov = params['h_fov']
        self.v_fov = params['v_fov']
        self.d_rot = params['d_rot']
        self.d_trans = params['d_trans']

        self.img_path = []
        self.lidar_path = []
        for drive in drives:
            cur_img_path = Path(base_path)/date/(date + '_drive_{:04d}_sync'.format(drive))/'image_02'/'data'
            cur_lidar_path = Path(base_path)/date/(date + '_drive_{:04d}_sync'.format(drive))/'velodyne_points'/'data'
            for i in range(len(list(cur_img_path.glob('*')))):
                self.img_path.append(str(cur_img_path/'{:010d}.png'.format(i)))
                self.lidar_path.append(str(cur_lidar_path/'{:010d}.bin'.format(i)))

        CAM02_PARAMS, VELO_PARAMS = calib.get_calib(date)
        self.cam_intrinsic = utils.get_intrinsic(CAM02_PARAMS['fx'], CAM02_PARAMS['fy'], CAM02_PARAMS['cx'], CAM02_PARAMS['cy'])
        self.velo_extrinsic = utils.get_extrinsic(VELO_PARAMS['rot'], VELO_PARAMS['trans'])

    def load_image(self, index):
        return cv2.imread(self.img_path[index])[:, :, ::-1]

    def load_lidar(self, index):
        return np.fromfile(self.lidar_path[index], dtype=np.float32).reshape(-1, 4)

    def get_projected_pts(self, index, extrinsic, img_shape, h_fov, v_fov):
        pcl = self.load_lidar(index)
        pcl_uv, pcl_z = utils.get_2D_lidar_projection(pcl, self.cam_intrinsic, extrinsic, h_fov, v_fov)
        outliers = utils.get_projection_outlier_idx(pcl_uv, img_shape)
        pcl_uv = pcl_uv[~outliers]
        pcl_z = pcl_z[~outliers]
        return pcl_uv, pcl_z

    def get_depth_image(self, index, extrinsic, img_shape, h_fov, v_fov):
        pcl_uv, pcl_z = self.get_projected_pts(index, extrinsic, img_shape, h_fov, v_fov)
        pcl_uv = pcl_uv.astype(np.uint32)
        pcl_z = pcl_z.reshape(-1, 1)
        depth_img = np.zeros((img_shape[0], img_shape[1], 1))
        depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
        return depth_img

    def __len__(self):
        return len(self.img_path)

    def get_decalibration(self):
        def get_rand():
            return np.random.rand() * 2 - 1

        d_roll = get_rand()*utils.degree_to_rad(self.d_rot)
        d_pitch = get_rand()*utils.degree_to_rad(self.d_rot)
        d_yaw = get_rand()*utils.degree_to_rad(self.d_rot)
        d_x = get_rand()*self.d_trans
        d_y = get_rand()*self.d_trans
        d_z = get_rand()*self.d_trans
        decalib_val = dict(
            d_rot_angle = [d_roll, d_pitch, d_yaw],
            d_trans = [d_x, d_y, d_z],
        )

        decalib_rot = utils.euler_to_rotmat(d_roll, d_pitch, d_yaw)
        decalib_trans = np.asarray([d_x, d_y, d_z]).reshape(3, 1)
        decalib_extrinsic = utils.get_extrinsic(decalib_rot, decalib_trans)

        return decalib_extrinsic, decalib_val

    def __getitem__(self, index):
        rgb_img = self.load_image(index)
        rgb_img = utils.mean_normalize_pts(rgb_img).astype('float32')

        decalib_extrinsic, _ = self.get_decalibration()
        decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(decalib_extrinsic)
        decalib_quat_real, decalib_quat_dual = utils.normalize_dual_quat(decalib_quat_real, decalib_quat_dual)

        init_extrinsic = utils.mult_extrinsic(self.velo_extrinsic, decalib_extrinsic)

        depth_img = self.get_depth_image(index, init_extrinsic, rgb_img.shape, self.h_fov, self.v_fov)
        depth_img = utils.mean_normalize_pts(depth_img).astype('float32')

        rgb_img = cv2.resize(rgb_img, (224, 224))
        depth_img = cv2.resize(depth_img, (224, 224))
        depth_img = depth_img[:, :, np.newaxis]

        decalib_quat_real = torch.from_numpy(decalib_quat_real)
        decalib_quat_dual = torch.from_numpy(decalib_quat_dual)
        rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1)
        depth_img = torch.from_numpy(depth_img).permute(2, 0, 1)

        sample = {}
        sample['rgb'] = rgb_img
        sample['depth'] = depth_img
        sample['decalib_real_gt'] = decalib_quat_real
        sample['decalib_dual_gt'] = decalib_quat_dual
        sample['init_extrinsic'] = init_extrinsic
        sample['index'] = index
        return sample
