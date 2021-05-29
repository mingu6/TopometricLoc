import importlib
import os
import os.path as path
import numpy as np

import yaml

from .settings import RESULTS_DIR


def import_localization(method_name):
    '''Import relevant localization object for a given method name'''
    localization = importlib.import_module(f".models.{method_name}", package='topometricloc')
    return localization.Localization


def load_params(method_name, fname=None):
    if fname:
        params_fname = fname
    else:
        params_fname = ("topometric" if method_name in ['topometric', 'nooff'] else method_name) + ".yaml"
    self_dirpath = path.dirname(path.abspath(__file__))
    params_path = path.abspath(path.join(self_dirpath, "params"))
    with open(path.join(params_path, params_fname), 'r') as f:
        params = yaml.safe_load(f)
    return params


def create_results_directory(args, query_traverse, method_name, exper):
    description = f"{args.reference_traverse}_{args.reference_filename[:-4]}_wd_{args.width}_" + \
            f"{query_traverse}_{args.query_filename[:-4]}_{method_name}_{args.descriptor}"
    results_dir = path.join(RESULTS_DIR, exper)
    os.makedirs(results_dir, exist_ok=True)
    trials = [int(p.split("_")[-1]) for p in os.listdir(results_dir) if "_".join(p.split("_")[:-1]) == description]
    trial_num = 1 if len(trials) <= 0 else max(trials) + 1
    results_path = path.join(results_dir, f"{description}_{trial_num}")
    os.makedirs(results_path)
    return results_path


def max_nonzero(mat, axis=None):
    '''Takes max of sparse matrix along an axis ignoring zeros'''
    eps = 1.
    min_val = mat.min()
    temp = mat.copy()
    temp.data -= min_val - eps  # raises all nonzero vals above zero if below
    max_vals = temp.max(axis=axis)
    if axis is not None:
        max_vals.data += min_val - eps  # sparse matrix case is axis provided
    else:
        max_vals += min_val - eps  # scalar case
    return max_vals



def logsumexp_nonzero(a, axis=None):
    '''logsumexp function for nonzero values along axis for sparse matrix'''
    eps = 1.  # ensure that no elements zeroed out in sparse matrix, making element disappear
    a_max = max_nonzero(a)
    temp = a.copy()
    temp.data = np.exp(a.data - a_max - eps)
    if axis is None:
        lse = np.log(temp.data.sum()) + a_max + eps
    else:
        sumexp = np.asarray(temp.sum(axis=axis)).squeeze()
        lse = np.log(sumexp) + a_max + eps
    return lse

def logsumexp_nonzero(a, axis=None, vec_bc_ind=None):
    '''logsumexp function for nonzero values along axis for sparse matrix'''
    eps = 1.  # ensure that no elements zeroed out in sparse matrix, making element disappear
    if axis is not None:
        a_max = max_nonzero(a, axis=axis).toarray().squeeze()
    else:
        a_max = max_nonzero(a, axis=None)
    temp = a.copy()
    if axis is None:
        temp.data = np.exp(a.data - a_max)
        lse = np.log(temp.data.sum()) + a_max
    elif axis == 0:
        temp = temp.tocsc()
        temp = sparse_nz_sum(temp, -a_max - eps, vec_bc_ind=vec_bc_ind)
        temp.data = np.exp(temp.data)
        sumexp = np.asarray(temp.sum(axis=axis)).squeeze()
        if np.any(sumexp == 0.):
            import pdb; pdb.set_trace()
        lse = np.log(sumexp) + a_max + eps
    elif axis == 1:
        temp = temp.tocsr()
        temp = sparse_nz_sum(temp, -a_max - eps, vec_bc_ind=vec_bc_ind)
        temp.data = np.exp(temp.data)
        sumexp = np.asarray(temp.sum(axis=axis)).squeeze()
        if np.any(sumexp == 0.):
            import pdb; pdb.set_trace()
        lse = np.log(sumexp) + a_max + eps
    return lse
def bc_vec_to_data_inds(mat):
    n_axes = len(mat.indptr)
    bc_vec = np.empty(len(mat.data), dtype=np.int32)
    for i in range(n_axes-1):
        bc_vec[mat.indptr[i]:mat.indptr[i+1]] = i
    return bc_vec


def sparse_nz_sum(mat, vec, vec_bc_ind=None):
    '''
    Add row/column vector to sparse matrix given broadcast indices.
    Args:
        mat: sparse matrix (n x m for csr or m x n for csc)
        vec: dense vector (len n)
        vec_bc_ind: (optional, None or len nz vector where nz is the number of
                non-zero elements of mat). Broadcast vector to size of data
                within mat. Has structure vec_bc_ind[indptr[i]:indptr[i+1]] = i,
                where indptr are the index pointers for mat.
    '''
    matf = mat.getformat()
    assert matf in ['csc', 'csr']
    if matf  == 'csr':
        assert mat.shape[0] == len(vec)
    elif matf == 'csc':
        assert mat.shape[1] == len(vec)
    if vec_bc_ind is None:
        vec_bc_ind = bc_vec_to_data_inds(mat)
    mat_vec_sum = mat.copy()
    mat_vec_sum.data += vec[vec_bc_ind]
    return mat_vec_sum


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


def pose_err_elementwise(poses1, poses2, degrees=False):
    assert poses1.shape == poses2.shape
    assert poses1.ndim == 2 and poses2.ndim == 2
    diff12 = poses1 - poses2
    t_err = np.linalg.norm(diff12[:, :2], axis=1)
    R_err = np.abs(diff12[:, -1])
    R_err[R_err > np.pi] = 2. * np.pi - R_err[R_err > np.pi]
    if degrees:
        R_err *= 180. / np.pi
    return t_err, R_err


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
