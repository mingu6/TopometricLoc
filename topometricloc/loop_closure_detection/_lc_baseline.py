import numpy as np

from .. import utils


def evaluate_proposal_error(proposals, query_gt):
    assert proposals.shape == query_gt.shape
    gt_errs_xy, gt_errs_rot = utils.pose_err_elementwise(query_gt, proposals, degrees=True)
    return gt_errs_xy, gt_errs_rot


def loop_closure_detection(localization, ref_map, query_global, odom_mu, odom_sigma):
    query_sims = ref_map.glb_des @ query_global.T
    dist = np.sqrt(2. - 2. * query_sims)
    min_idx = np.argmin(dist, axis=0)
    pose_ests = ref_map.gt_poses[min_idx]
    scores = -np.min(dist, axis=0)
    return scores, pose_ests
