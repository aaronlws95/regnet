import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import src.utils as utils
import src.visualize as vis
from src.dataset import Kitti_Dataset

# Setup
dataset_params = {
    'base_path': Path('data')/'KITTI_SMALL',
    'date': '2011_09_26',
    'drives': [5],
    'h_fov': (-90, 90),
    'v_fov': (-24.9, 2.0),
    'd_rot': 1,
    'd_trans': 0.1,
    'fixed_decalib': True,
}
dataset = Kitti_Dataset(dataset_params)

i = 0

# Show Image
img = dataset.load_image(i)
pcl_uv, pcl_z = dataset.get_projected_pts(i, dataset.velo_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
img_projected = vis.get_projected_img(pcl_uv, pcl_z, img)
# img_projected = cv2.resize(img_projected, (224, 244))
# vis.show_img(img_projected)

# Depth map without FOV
# pcl = dataset.load_lidar(i)
# pcl_xyz = np.hstack((pcl[:, :3], np.ones((pcl.shape[0], 1)))).T
# pcl_xyz = dataset.velo_extrinsic @ pcl_xyz
# pcl_xyz = dataset.cam_intrinsic @ pcl_xyz
# pcl_xyz = pcl_xyz.T
# pcl_z = pcl_xyz[:, 2]
# pcl_xyz = pcl_xyz / pcl_xyz[:, 2, None]
# pcl_uv = pcl_xyz[:, :2]
# pcl_u = pcl_xyz[:, 0]
# pcl_v = pcl_xyz[:, 1]

# height = img.shape[0]
# width = img.shape[1]
# dimg = np.zeros((height, width, 1))
# mask = (pcl_u > 0) & (pcl_u < width) & (pcl_v > 0) & (pcl_v < height) & (pcl_z > 0)
# ymask, xmask = pcl_v[mask].astype(int), pcl_u[mask].astype(int)
# dimg[ymask, xmask, 0] = pcl_z[mask]

# lidarimg = np.zeros((height, width, 1))
# lidarimg[ymask, xmask, 0] = pcl[mask, 3]

# # vis.show_img(dimg.squeeze())
# pcl_uv2, pcl_z2 = dataset.get_projected_pts(i, dataset.velo_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)

# gt_depth_img = dataset.get_depth_image(i, dataset.velo_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# gt_depth_img = utils.mean_normalize_pts(gt_depth_img).astype('float32').squeeze()
# # gt_depth_img = cv2.resize(gt_depth_img, (dataset.img_size, dataset.img_size))
# # vis.show_img(gt_depth_img)

# print(pcl_z2.shape, pcl_z[mask].shape)

# fig, ax = plt.subplots(3, 1)
# ax[0].imshow(lidarimg.squeeze())
# ax[1].imshow(dimg.squeeze())
# ax[2].imshow(gt_depth_img)
# plt.show()

# # dimg[1, ymask, xmask] = lidar[mask,3]

# img_projected = vis.get_projected_img(pcl_uv[mask], pcl_z[mask], img)
# img_projected2 = vis.get_projected_img(pcl_uv2, pcl_z2, img)

# fig, ax = plt.subplots(2, 1)
# ax[0].imshow(img_projected)
# ax[1].imshow(img_projected2)
# plt.show()


# Rotation Matrix <-> Euler
# rotmat = utils.get_rotmat_from_extrinsic(dataset.velo_extrinsic)
# [roll, pitch, yaw] = utils.rotmat_to_euler(rotmat)
# print(utils.rad_to_degree(roll), utils.rad_to_degree(pitch), utils.rad_to_degree(yaw))
# new_rotmat = utils.euler_to_rotmat(roll, pitch, yaw)
# new_extrinsic = utils.get_extrinsic(new_rotmat, dataset.VELO_PARAMS['trans'])
# pcl_uv, pcl_z = dataset.get_projected_pts(i, new_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# vis.show_projection(pcl_uv, pcl_z, img)

# Rotation Matrix <-> Quaternion
# rotmat = utils.get_rotmat_from_extrinsic(dataset.velo_extrinsic)
# quat = utils.rotmat_to_quat(rotmat)
# new_rotmat = utils.quat_to_rotmat(*quat)
# new_extrinsic = utils.get_extrinsic(new_rotmat, dataset.VELO_PARAMS['trans'])
# pcl_uv, pcl_z = dataset.get_projected_pts(i, new_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# vis.show_projection(pcl_uv, pcl_z, img)

# Extrinsic <-> Dual Quaternion
# real_quat, dual_quat = utils.extrinsic_to_dual_quat(dataset.velo_extrinsic)
# new_extrinsic = utils.dual_quat_to_extrinsic(real_quat, dual_quat)
# pcl_uv, pcl_z = dataset.get_projected_pts(i, new_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# vis.show_projection(pcl_uv, pcl_z, img)

# Decalibration

# decalib_extrinsic, decalib_val = dataset.get_decalibration()
# decalib_rotangles = np.asarray([utils.rad_to_degree(a) for a in decalib_val['d_rot_angle']])
# decalib_trans = decalib_val['d_trans']

# print('Decalib Rot', decalib_rotangles)
# print('Decalib Trans', decalib_trans)

# init_extrinsic = utils.mult_extrinsic(dataset.velo_extrinsic, decalib_extrinsic)
# init_rotmat = utils.get_rotmat_from_extrinsic(init_extrinsic)
# init_trans = utils.get_trans_from_extrinsic(init_extrinsic)
# init_rotangles = utils.rotmat_to_euler(init_rotmat, out='deg')

# print('Initial Rotation', init_rotangles)
# print('Initial Trans', init_trans)

# true_rotangles = utils.rotmat_to_euler(utils.get_rotmat_from_extrinsic(dataset.velo_extrinsic), out='deg')
# true_trans =  utils.get_trans_from_extrinsic(dataset.velo_extrinsic)

# print('True Rotation', true_rotangles)
# print('True Translation', true_trans)

# inv_decalib_extrinsic = utils.inv_extrinsic(decalib_extrinsic)
# reprojected_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_decalib_extrinsic)
# reprojected_rotmat = utils.get_rotmat_from_extrinsic(reprojected_extrinsic)
# reprojected_rotangles = utils.rotmat_to_euler(reprojected_rotmat, out='deg')
# reprojected_trans = utils.get_trans_from_extrinsic(reprojected_extrinsic)

# print('Reprojected Rotation', reprojected_rotangles)
# print('Reprojected Translation', reprojected_trans)

# decalib_quat_real, decalib_quat_dual = utils.extrinsic_to_dual_quat(decalib_extrinsic)
# w = decalib_quat_real[0]
# x = decalib_quat_real[1]
# y = decalib_quat_real[2]
# z = decalib_quat_real[3]
# print(np.sqrt(w * w + x * x + y * y + z * z) + 1e-10)
# print('Decalib Quat Real', decalib_quat_real)
# print('Decalib Quat Dual', decalib_quat_dual)

# quat_decalib_extrinsic = utils.dual_quat_to_extrinsic(decalib_quat_real, decalib_quat_dual)
# inv_quat_decalib_extrinsic = utils.inv_extrinsic(quat_decalib_extrinsic)
# quat_reprojected_extrinsic = utils.mult_extrinsic(init_extrinsic, inv_quat_decalib_extrinsic)
# quat_reprojected_rotangles = utils.rotmat_to_euler(quat_reprojected_extrinsic, out='deg')
# quat_reprojected_trans = utils.get_trans_from_extrinsic(quat_reprojected_extrinsic)

# print('Quat Reprojected Rotation', quat_reprojected_rotangles)
# print('Quat Reprojected Translation', quat_reprojected_trans)

# pcl_uv, pcl_z = dataset.get_projected_pts(i, init_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# vis.show_projection(pcl_uv, pcl_z, img)

# pcl_uv, pcl_z = dataset.get_projected_pts(i, reprojected_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# vis.show_projection(pcl_uv, pcl_z, img)

# pcl_uv, pcl_z = dataset.get_projected_pts(i, quat_reprojected_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# vis.show_projection(pcl_uv, pcl_z, img)

# depth_img = dataset.get_depth_image(i, dataset.velo_extrinsic, img.shape, dataset.h_fov, dataset.v_fov).squeeze()
# depth_img = utils.mean_normalize_pts(depth_img).astype('float32')

# depth_img = cv2.resize(depth_img, (224, 244))
# vis.show_img(depth_img)

# Checking stuff
# new_rotmat = utils.euler_to_rotmat(utils.degree_to_rad(95.31148369), utils.degree_to_rad(-88.83698953), utils.degree_to_rad(-6.16228829))
# new_extrinsic = utils.get_extrinsic(new_rotmat, utils.get_trans_from_extrinsic(dataset.velo_extrinsic).reshape(3, 1))

# inv_extrinsic = utils.inv_extrinsic(new_extrinsic)
# diff_extrinsic = utils.mult_extrinsic(dataset.velo_extrinsic, inv_extrinsic)
# diff_rotmat = utils.get_rotmat_from_extrinsic(diff_extrinsic)
# diff_trans = utils.get_trans_from_extrinsic(diff_extrinsic)

# print(diff_trans)

# diff_euler = utils.rotmat_to_euler(diff_rotmat, out='deg')

# print(diff_euler)

# [roll, pitch, yaw] = utils.rotmat_to_euler(new_rotmat)
# # print(utils.rad_to_degree(roll), utils.rad_to_degree(pitch), utils.rad_to_degree(yaw))

# pcl_uv, pcl_z = dataset.get_projected_pts(i, new_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# pred_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

# pcl_uv, pcl_z = dataset.get_projected_pts(i, dataset.velo_extrinsic, img.shape, dataset.h_fov, dataset.v_fov)
# gt_projected_img = vis.get_projected_img(pcl_uv, pcl_z, img)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2, 1)
# ax[0].imshow(pred_projected_img)
# ax[1].imshow(gt_projected_img)
# plt.show()

# Pred Decalib Quaternion Real Part:
# Pred Decalib Quaternion Dual Part:
# GT Decalib Quaternion Real Part: [ 9.99951528e-01  4.77371930e-04 -7.74812900e-03 -6.05641086e-03]
# GT Decalib Quaternion Dual Part: [-2.89888490e-03  1.98916358e-03  1.90387945e-02 -6.94438065e-06]
# Pred Extrinsic [[ 0.02009489 -0.9954847  -0.0148054   0.00984411]
#  [-0.00216962  0.01476459 -0.9956858  -0.08342741]
#  [ 0.99559258  0.02012489 -0.001871   -0.27754566]]
# GT Extrinsic [[ 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03]
#  [ 1.480249e-02  7.280733e-04 -9.998902e-01 -7.631618e-02]
#  [ 9.998621e-01  7.523790e-03  1.480755e-02 -2.717806e-01]]
# [ 95.31148369 -88.83698953  -6.16228829] [ 26.93535073 -89.04830974  63.02612693]
# Roll Error 68.37613296167677
# Pitch Error 0.2113202092587727
# Yaw Error 69.18841522057265
# X Error 0.013913877973204671
# Y Error 0.007111231848944313
# Z Error 0.00576506226206186