import math
import time
import torch
import numpy as np

from utils.paramUtil import FACE_JOINT_INDX, FID_L, FID_R, FACE_JOINT_INDX, HML_KINEMATIC_CHAIN, HML_RAW_OFFSETS
from utils.quaternion import *
from utils.rotation_conversions import *
from utils.skeleton import Skeleton, uniform_skeleton

### MOTION NORMZALIZERS ###

class MotionNormalizer():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x
    
    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x

class MotionNormalizerHML3D():
    def __init__(self):
        mean = np.load("./data/HumanML3D/mean_ih_new.npy")
        std = np.load("./data/HumanML3D/std_ih_new.npy")
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x

    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x


class MotionNormalizerTorch():
    def __init__(self):
        mean = np.load("./data/global_mean.npy")
        std = np.load("./data/global_std.npy")
        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()

    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x


class MotionNormalizerTorchHML3D():
    def __init__(self):
        mean = np.load("./data/HumanML3D/mean_ih_new.npy")
        std = np.load("./data/HumanML3D/std_ih_new.npy")
        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x
    

### MOTION PROCESSORS ###

TRANS_MATRIX = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


def process_motion_interhuman(motion, feet_thre, prev_frames, n_joints, flip=True, skel=False):
    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)
    rotations = motion[:, n_joints*3:]

    if skel:
        positions = uniform_skeleton(positions)

    if flip:
        positions = np.einsum("mn, tjn->tjm", TRANS_MATRIX, positions)

    # Put on Floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    # XZ at origin
    root_pos_init = positions[prev_frames]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # All initially face Z+
    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init_for_all, positions)

    # Get Foot Contacts
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, FID_L, 0] - positions[:-1, FID_L, 0]) ** 2
        feet_l_y = (positions[1:, FID_L, 1] - positions[:-1, FID_L, 1]) ** 2
        feet_l_z = (positions[1:, FID_L, 2] - positions[:-1, FID_L, 2]) ** 2
        feet_l_h = positions[:-1,FID_L,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, FID_R, 0] - positions[:-1, FID_R, 0]) ** 2
        feet_r_y = (positions[1:, FID_R, 1] - positions[:-1, FID_R, 1]) ** 2
        feet_r_z = (positions[1:, FID_R, 2] - positions[:-1, FID_R, 2]) ** 2
        feet_r_h = positions[:-1,FID_R,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)

    # Get Joint Rotation Representation
    rot_data = rotations

    # Get Joint Rotation Invariant Position Represention
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    # Generate final motion vector
    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]

### UTILS ###

def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)



def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def rigid_transform(relative, data):
    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data