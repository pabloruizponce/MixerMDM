import torch
import numpy as np
import utils.rotation_conversions as rc


from aitviewer.renderables.lines import Lines
from utils.paramUtil import FACE_JOINT_INDX
from utils.quaternion import qbetween, qbetween_np, qrot


def ih_to_smpl(motion):
    """
    Convert motion from IH format to SMPL format for batches.

    Parameters:
    motion (torch.Tensor): A tensor of shape (batch_size, num_features) representing the motion data.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, new_num_features) representing the converted motion data.
    """
    batch_size = motion.shape[0]

    # Get poses
    poses = motion[:, :, 22*6:22*6+21*6]
    poses = poses.reshape(batch_size, -1, 21, 6).float()

    # Fix motion representation to properly visualization
    poses = rc.rotation_6d_to_matrix(poses)
    poses = rc.matrix_to_axis_angle(poses)
    poses = poses * -1

    poses = poses.reshape(batch_size, -1, 21*3)
    
    # Add the 2 additional joints from the hands being 0
    zero_padding = torch.zeros([batch_size, poses.shape[1], 6], device=poses.device)
    poses = torch.cat([poses, zero_padding], axis=2)

    new_motion = torch.cat([motion[:, :, :22*6], poses, motion[:, :, -4:]], axis=2)
    return new_motion

def smpl_to_ih(motion):
    """
    Convert motion from SMPL format to IH format for batches.

    Parameters:
    motion (torch.Tensor): A tensor of shape (batch_size, num_features) representing the motion data.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, new_num_features) representing the converted motion data.
    """
    batch_size = motion.shape[0]

    # Get poses
    poses = motion[:, :, 22*6:22*6+23*3]
    poses = poses.reshape(batch_size, -1, 23, 3).float()

    # Fix motion representation to properly visualization
    poses = poses * -1
    poses = rc.axis_angle_to_matrix(poses)
    poses = rc.matrix_to_rotation_6d(poses)
    poses = poses.reshape(batch_size, -1, 23*6)

    # Remove the 2 additional joints from the hands being 0
    poses = poses[:, :, :-6*2]
    
    new_motion = torch.cat([motion[:, :, :22*6], poses, motion[:, :, -4:]], axis=2)
    return new_motion

def align_trajectories(t1, t2, mask=None):
    """
    Aligns trajectories t1 and t2 for batches.

    Parameters:
    t1 (torch.Tensor): A tensor of shape (batch_size, sequence_length, 3) representing trajectory 1.
    t2 (torch.Tensor): A tensor of shape (batch_size, sequence_length, 3) representing trajectory 2.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, sequence_length, 22, 4) representing aligned rotations.
    """
    batch_size = t1.shape[0]

    # Get vector with the initial and end position of the trajectory 1 and 2
    if mask is None:
        v1 = t1[:, -1] - t1[:, 0]
        v2 = t2[:, -1] - t2[:, 0]
    else:
        lenghts = mask.squeeze().sum(dim=1).int()
        v1 = t1[torch.arange(batch_size), lenghts-1] - t1[:, 0]
        v2 = t2[torch.arange(batch_size), lenghts-1] - t2[:, 0]

    # Ignore the y component on the alignment
    v1[:, 1] = 0
    v2[:, 1] = 0

    # Normalize vectors to have unit length
    v1 = v1 / torch.sqrt((v1 ** 2).sum(dim=1, keepdim=True) + 1e-8)
    v2 = v2 / torch.sqrt((v2 ** 2).sum(dim=1, keepdim=True) + 1e-8)

    # Get the rotation quaternion between the two vectors
    rot_quat = qbetween(v2, v1)
    
    # Create a batch of rotation quaternions
    root_quat_for_all = torch.ones(t2.shape[:-1] + (22,) + (4,), device=t2.device) * rot_quat.unsqueeze(1).unsqueeze(2)
    
    return root_quat_for_all


def align_motions(motion1, motion2, mask=None):
    """
    Aligns two motion sequences by adjusting positions and rotations.

    Args:
        motion1: Tensor representing the first motion sequence.
        motion2: Tensor representing the second motion sequence.

    Returns:
        Two tensors representing the aligned motion sequences.
    """
    # Make copies of input motions
    motion1 = motion1.clone()
    motion2 = motion2.clone()

    B = motion1.shape[0]

    # Extract positions, velocities, and rotations from both motions
    positions1 = motion1[..., :22*3].reshape(B, -1, 22, 3)
    velocities1 = motion1[..., 22*3:22*6].reshape(B, -1, 22, 3)
    rotations1 = motion1[..., 22*6:22*6+23*3].reshape(B, -1, 23, 3)

    positions2 = motion2[..., :22*3].reshape(B, -1, 22, 3)
    velocities2 = motion2[..., 22*3:22*6].reshape(B, -1, 22, 3)
    rotations2 = motion2[..., 22*6:22*6+23*3].reshape(B, -1, 23, 3)

    # Align initial positions of the two motions
    positions2 = positions2 + (positions1[:,0,0] - positions2[:,0,0]).unsqueeze(1).unsqueeze(1).expand(-1,positions2.shape[1],22,-1)

    # Align rotations using trajectory alignment
    alignment = align_trajectories(positions1[:,:,0], positions2[:,:,0], mask)

    # Apply rotation alignment to positions and velocities
    positions2 = qrot(alignment, positions2)
    
    # Re-align positions after rotation
    positions2 = positions2 + (positions1[:,0,0] - positions2[:,0,0]).unsqueeze(1).unsqueeze(1).expand(-1,positions2.shape[1],22,-1)

    velocities2 = qrot(alignment, velocities2)

    # Reconstruct the aligned second motion sequence
    motion2 = torch.cat(
        [
            positions2.reshape(B, -1, 22*3), 
            velocities2.reshape(B, -1, 22*3), 
            rotations2.reshape(B, -1, 23*3)
        ], 
        axis=-1
    )

    return motion1, motion2


def center_motion(motion):
    """
    Centers motion data for batches.

    Parameters:
    motion (torch.Tensor): A tensor of shape (batch_size, num_frames, num_features) representing motion data.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, num_frames, num_features) representing centered motion data.
    """
    motion = motion.clone()
    
    B = motion.shape[0]

    # Extract the positions, velocities and rotations
    positions = motion[:, :, :22*3].reshape(B, -1, 22, 3)
    velocities = motion[:, :, 22*3:22*6].reshape(B, -1, 22, 3)
    rotations = motion[:, :, 22*6:22*6+23*3].reshape(B, -1, 23, 3)

    # Put on Floor
    floor_height = positions.min(dim=1).values.min(dim=1).values[:, 1]
    positions = positions.clone()  # Avoid in-place operation
    positions[:, :, :, 1] -= floor_height.unsqueeze(-1).unsqueeze(-1)

    # XZ at origin
    root_pos_init = positions[:, 0]
    root_pose_init_xz = root_pos_init[:, 0] * torch.tensor([1, 0, 1], device=positions.device)
    positions = positions - root_pose_init_xz.unsqueeze(1).unsqueeze(2)

    # All initially face Z+
    r_hip, l_hip = FACE_JOINT_INDX[:2]  # Assuming FACE_JOINT_INDX is defined elsewhere
    across = root_pos_init[:, r_hip] - root_pos_init[:, l_hip]
    
    # Normalize across vector without in-place operation
    across_norm = torch.sqrt((across ** 2).sum(dim=-1)).unsqueeze(-1).float()
    across = across / across_norm
    
    forward_init = torch.cross(torch.tensor([0, 1, 0], device=positions.device).expand(B,-1).float(), across.float())
    
    # Normalize forward_init vector without in-place operation
    forward_norm = torch.sqrt((forward_init ** 2).sum(dim=-1)).unsqueeze(-1)
    forward_init = forward_init / forward_norm
    
    target = torch.tensor([0, 0, 1], device=positions.device).expand(B, -1)
    
    root_quat_init = qbetween(forward_init.float(), target.float())    
    root_quat_init_for_all = root_quat_init.unsqueeze(1).unsqueeze(1).expand(-1, positions.shape[1], positions.shape[2], -1)

    # Rotate the positions and velocities
    positions = qrot(root_quat_init_for_all.float(), positions.float())
    velocities = qrot(root_quat_init_for_all.float(), velocities.float())

    motion_centered = torch.cat(
        [
            positions.reshape(B, -1, 22*3), 
            velocities.reshape(B, -1, 22*3), 
            rotations.reshape(B, -1, 23*3)
        ], 
        axis=-1
    )
    
    return motion_centered

## Only used for visualization not needed to be adapted to batches

def get_lines_trajectory(trajectory, full=False):
    # Get number of frames
    n_frames = trajectory.shape[0]
    
    # Calculate lines begining and end
    lines = np.zeros(((n_frames-1),(n_frames-1)* 2, 3))
    lines[:,::2] = trajectory[:-1]
    lines[:,1::2] = trajectory[1:]

    # Only make visible the lines before the actual frame
    if not full:
        for i in range(n_frames-1):
            lines[i, 2*i+2:] = 0

    r_lines = Lines(lines, mode="lines")
    
    return r_lines

def extract_smpl(motion):
    # Extract the positions, velocities and rotations
    positions = motion[:, :22*3].reshape(-1, 22, 3)
    rotations = motion[:, 22*6:22*6+23*3].reshape(-1, 23, 3)

    poses = rotations.reshape(-1, 23*3)
    trans = positions[:,0]
    root_poses = get_root_pos(positions)

    return poses, trans, root_poses

def get_root_pos(positions):
    positions = positions.copy().reshape(-1, 22, 3)
    r_hip, l_hip = FACE_JOINT_INDX[:2]
    across = positions[:, r_hip] - positions[:, l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1) + 1e-8)[..., np.newaxis]

    # Get perpendicular vectors to the across vector
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # Normalize the forward vector to have unit length
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1) + 1e-8)[..., np.newaxis]

    # Get the rotation quaternion between the forward vector and the origin
    origin = np.array([[0, 0, 1]])
    root_pose_quat = qbetween_np(origin, forward)
    root_pose_quat = np.nan_to_num(root_pose_quat)

    root_pose = rc.quaternion_to_axis_angle(torch.Tensor(root_pose_quat))    
    return root_pose.detach().cpu().numpy()