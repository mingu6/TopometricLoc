import numpy as np


def pose_diff(poses1, poses2):
    '''
    Computes difference in poses (translation and rotation) between two sets of 3dof poses.
    Accepts single poses for either argument, will broadcast appropriately.
    Args:
        poses1 (np array Nx3 or 3): Set of (or single) pose(s)
        poses2 (np array Mx3 or 3): Set of (or single) pose(s)
    Returns:
        diff12 (np array NxMx3 or Nx3 or Mx3 or 3): Pose differences.
    '''
    assert poses1.shape[-1] == 3
    assert poses2.shape[-1] == 3
    poses1 = np.atleast_2d(poses1)
    poses2 = np.atleast_2d(poses2)
    diff12 = np.empty((poses1.shape[0], poses2.shape[0], 3))
    for i, p1 in enumerate(poses1):
        d1 = p1[None, :] - poses2
        diff12[i, ...] = d1
    diff12 = np.squeeze(diff12)
    return diff12


def pose_err(poses1, poses2, degrees=False):
    """
    Computes error in poses (translation and rotation) between two sets of 3dof poses.
    Accepts single poses for either argument, will broadcast appropriately.
    Args:
        poses1 (np array Nx3 or 3): Set of (or single) pose(s)
        poses2 (np array Mx3 or 3): Set of (or single) pose(s)
        degrees (boolean, optional): Return angular difference in degrees if True
    Returns:
        diff12 (np array NxMx3 or Nx3 or Mx3 or 3): Pose differences.
    """
    diff12 = pose_diff(poses1, poses2)
    t_err = np.linalg.norm(diff12[..., :2], axis=-1)
    R_err = np.abs(diff12[..., -1])
    # correct angular difference if too high/low
    if R_err.shape:  # handle single t_err, R_err values separately
        R_err[R_err > np.pi] = 2. * np.pi - R_err[R_err > np.pi]
    else:
        R_err = 2. * np.pi - R_err if R_err > np.pi else R_err
    if degrees:
        R_err *= 180. / np.pi
    return t_err, R_err
