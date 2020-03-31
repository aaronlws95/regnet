from pathlib import Path
import cv2
import numpy as np

def calibration_error(e1, e2):
    inv_e2 = inv_extrinsic(e2)
    err_e = mult_extrinsic(e1, inv_e2)
    err_rotmat = get_rotmat_from_extrinsic(err_e)
    err_trans = get_trans_from_extrinsic(err_e)
    err_euler = rotmat_to_euler(err_rotmat, out='deg')

    err_euler = [np.abs(i) for i in err_euler]
    err_trans = [np.abs(i) for i in err_trans]

    return err_euler[0], err_euler[1], err_euler[2], err_trans[0], err_trans[1], err_trans[2]

def mean_squared_error(p1, p2):
    return np.mean((p1 - p2)**2)

def euclidean_distance(pts):
    return np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2).reshape(-1, 1)

def degree_to_rad(theta):
    return theta * np.pi / 180

def rad_to_degree(theta):
    return theta * 180 / np.pi

def get_fov_range_idx(n, m, fov_range):
    return np.logical_and(np.arctan2(n, m) < degree_to_rad(fov_range[1]), np.arctan2(n, m) >  degree_to_rad(fov_range[0]))

def max_normalize_pts(pts):
    return (pts - np.min(pts)) / (np.max(pts) - np.min(pts) + 1e-10)

def mean_normalize_pts(pts):
    return (pts - np.mean(pts)) / (np.std(pts) + 1e-10)

def filter_pts_by_fov(pts, h_fov, v_fov):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    d = np.sqrt(x**2 + y**2 + z**2)
    h_pts_idx = get_fov_range_idx(y, x, h_fov)
    v_pts_idx = get_fov_range_idx(z, d, v_fov)
    idx = np.logical_and(h_pts_idx, v_pts_idx)
    return pts[idx]

def get_projection_outlier_idx(pts, img_shape):
    u_outliers = np.logical_or(pts[:, 0] < 0, pts[:, 0] > img_shape[1])
    v_outliers = np.logical_or(pts[:, 1] < 0, pts[:, 1] > img_shape[0])
    outliers = np.logical_or(u_outliers, v_outliers)
    return outliers

def get_2D_lidar_projection_fov(pcl, cam_intrinsic, velo_extrinsic, h_fov, v_fov):
    filter_pcl = filter_pts_by_fov(pcl, h_fov, v_fov)
    pcl_xyz = np.hstack((filter_pcl[:, :3], np.ones((filter_pcl.shape[0],1)))).T
    pcl_xyz = velo_extrinsic @ pcl_xyz
    pcl_xyz = cam_intrinsic @ pcl_xyz
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / pcl_xyz[:, 2, None]
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z

def get_2D_lidar_projection(pcl, cam_intrinsic, velo_extrinsic):
    pcl_xyz = np.hstack((pcl[:, :3], np.ones((pcl.shape[0],1)))).T
    pcl_xyz = velo_extrinsic @ pcl_xyz
    pcl_xyz = cam_intrinsic @ pcl_xyz
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / pcl_xyz[:, 2, None]
    pcl_uv = pcl_xyz[:, :2]
    return pcl_uv, pcl_z

def rotmat_to_euler(rotmat, out='rad') :
    sy = np.sqrt(rotmat[0,0] * rotmat[0,0] +  rotmat[1,0] * rotmat[1,0])
    singular = sy < 1e-6

    if  not singular :
        roll = np.arctan2(rotmat[2, 1] , rotmat[2, 2])
        pitch = np.arctan2(-rotmat[2, 0], sy)
        yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])
    else :
        roll = np.arctan2(-rotmat[1, 2], rotmat[1, 1])
        pitch = np.arctan2(-rotmat[2, 0], sy)
        yaw = 0

    if out == 'rad':
        return np.asarray([roll, pitch, yaw])
    elif out == 'deg':
        return np.asarray([rad_to_degree(roll),
                           rad_to_degree(pitch),
                           rad_to_degree(yaw)])

def euler_to_rotmat(roll, pitch, yaw):
    rot_x = [[1, 0, 0],
             [0, np.cos(roll), -np.sin(roll)],
             [0, np.sin(roll), np.cos(roll)]]
    rot_x = np.asarray(rot_x)

    rot_y = [[np.cos(pitch), 0, np.sin(pitch)],
             [0, 1, 0],
             [-np.sin(pitch), 0, np.cos(pitch)]]
    rot_y = np.asarray(rot_y)

    rot_z = [[np.cos(yaw), -np.sin(yaw), 0],
             [np.sin(yaw), np.cos(yaw), 0],
             [0, 0, 1]]
    rot_z = np.asarray(rot_z)

    return rot_z @ rot_y @ rot_x

def get_intrinsic(fx, fy, cx, cy):
    return np.asarray([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])

def get_extrinsic(rot, trans):
    return np.hstack((rot, trans))

def rotmat_to_quat(rotmat):
    tr = rotmat[0, 0] + rotmat[1, 1] + rotmat[2, 2]

    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (rotmat[2, 1] - rotmat[1, 2]) / s
        qy = (rotmat[0, 2] - rotmat[2, 0]) / s
        qz = (rotmat[1, 0] - rotmat[0, 1]) / s
    elif rotmat[0, 0] > rotmat[1, 1] and rotmat[0, 0] > rotmat[2, 2]:
        s = sqrt(1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2]) * 2
        qw = (rotmat[2, 1] - rotmat[1, 2]) / s
        qx = 0.25 * s
        qy = (rotmat[0, 1] + rotmat[1, 0]) / s
        qz = (rotmat[0, 2] + rotmat[2, 0]) / s
    elif rotmat[1, 1] > rotmat[2, 2]:
        s = sqrt(1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2]) * 2
        qw = (rotmat[0, 2] - rotmat[2, 0]) / s
        qx = (rotmat[0, 1] + rotmat[1, 0]) / s
        qy =  0.25 * s
        qz = (rotmat[1, 2] + rotmat[2, 1]) / s
    else:
        s = sqrt(1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1]) * 2
        qw = (rotmat[1, 0] - rotmat[0, 1]) / s
        qx = (rotmat[0, 2] + rotmat[2, 0]) / s
        qy = (rotmat[1, 2] + rotmat[2, 1]) / s
        qz =  0.25 * s

    return np.asarray([qw, qx, qy, qz])

def quat_to_rotmat(qw, qx, qy, qz):
    r00 = 1 - 2 * qy**2 - 2 * qz**2
    r01 = 2 * qx*qy - 2 * qz * qw
    r02 = 2 * qx*qz + 2 * qy * qw

    r10 = 2 * qx * qy + 2 * qz * qw
    r11 = 1 - 2 * qx**2 - 2 * qz**2
    r12 = 2 * qy * qz - 2 * qx * qw

    r20 = 2 * qx * qz - 2 * qy * qw
    r21 = 2 * qy * qz + 2 * qx * qw
    r22 = 1 - 2 * qx**2 - 2 * qy**2

    rotmat = [[r00, r01, r02],
              [r10, r11, r12],
              [r20, r21, r22]]

    return np.asarray(rotmat)

def get_rotmat_from_extrinsic(extrinsic):
    return extrinsic[:3, :3]

def get_trans_from_extrinsic(extrinsic):
    return extrinsic[:3, 3]

def quat_mult(q1, q2):
    qw = -q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3] + q1[0] * q2[0]
    qx =  q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1]
    qy = -q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2]
    qz =  q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0] + q1[0] * q2[3]
    return np.asarray([qw, qx, qy, qz])

def conj_quat(q):
    return np.asarray([q[0], -q[1], -q[2], -q[3]])

def extrinsic_to_dual_quat(extrinsic):
    rotmat = get_rotmat_from_extrinsic(extrinsic)
    trans = get_trans_from_extrinsic(extrinsic)
    trans = np.asarray([trans[0], trans[1], trans[2], 0])
    real_quat = rotmat_to_quat(rotmat)
    dual_quat = quat_mult(trans, real_quat) * 0.5
    return real_quat, dual_quat

def dual_quat_to_extrinsic(real_quat, dual_quat):
    w = real_quat[0]
    x = real_quat[1]
    y = real_quat[2]
    z = real_quat[3]

    r00 = w * w + x * x - y * y - z*z
    r01 = 2 * x * y - 2 * w * z
    r02 = 2 * x * z + 2 * w * y

    r10 = 2 * x * y + 2 * w * z
    r11 = w * w + y * y - x * x - z * z
    r12 = 2 * y*z - 2 * w * x

    r20 = 2 * x * z - 2 * w * y
    r21 = 2 * y * z + 2 * w * x
    r22 = w * w + z * z - x * x - y*y

    t = quat_mult(2 * dual_quat, conj_quat(real_quat))

    rot = np.asarray([[r00, r01, r02],
                      [r10, r11, r12],
                      [r20, r21, r22]])

    trans = np.asarray([t[0], t[1], t[2]]).reshape(3, 1)
    return get_extrinsic(rot, trans)

def normalize_dual_quat(real_quat, dual_quat):
    w = real_quat[0]
    x = real_quat[1]
    y = real_quat[2]
    z = real_quat[3]
    mag = np.sqrt(w * w + x * x + y * y + z * z) + 1e-10
    return real_quat / mag, dual_quat / mag

def mult_extrinsic(m1, m2):
    mult = np.vstack((m1, [0, 0, 0, 1])) @ np.vstack((m2, [0, 0, 0, 1]))
    return mult[:3, :]

def inv_extrinsic(m):
    return np.linalg.inv(np.vstack((m, [0, 0, 0, 1])))[:3, :]
