from tqdm import trange, tqdm
import numpy as np

from scipy.special import logsumexp

from .. import utils


def forward_update(logalpha, log_trans_mat, log_lhood, data_bc_inds=None):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    result = np.empty(log_trans_mat.shape[0])
    # ensure csc format so indices are extracted properly
    mat = log_trans_mat.tocsc()
    indptr = mat.indptr
    for i in range(mat.shape[0]):
        indptr = mat.indptr
        indices = mat.indices[indptr[i]:indptr[i+1]]
        # extract relevant column vector entries
        col = logalpha[indices]
        result[i] = logsumexp(col + mat.data[indptr[i]:indptr[i+1]])
    return result + log_lhood

def forward_update(logalpha, log_trans_mat, log_lhood, data_bc_inds=None):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    log_motion_update = utils.sparse_nz_sum(log_trans_mat.tocsr(), logalpha, vec_bc_ind=data_bc_inds)
    log_motion_update = utils.logsumexp_nonzero(log_motion_update, axis=0)
    log_alpha_update = log_motion_update + log_lhood
    return log_alpha_update

def backward_update(logbeta, log_trans_mat, log_lhood, data_bc_inds=None):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    result = np.empty(log_trans_mat.shape[1])
    # ensure csr format so indices are extracted properly
    mat = log_trans_mat.tocsr()
    indptr = mat.indptr
    for i in range(mat.shape[1]):
        indptr = mat.indptr
        indices = mat.indices[indptr[i]:indptr[i+1]]
        # extract relevant column vector entries
        beta_row = logbeta[indices]
        lhood_row = log_lhood[indices]
        result[i] = logsumexp(beta_row + lhood_row + mat.data[indptr[i]:indptr[i+1]])
    return result


def backward_update(logbeta, log_trans_mat, log_lhood, data_bc_inds=None):
    """
    Numerically stable version of update using logsumexp. Assumes
    all inputs are in log-space already.
    """
    log_update = utils.sparse_nz_sum(log_trans_mat.tocsc(), logbeta + log_lhood, vec_bc_ind=data_bc_inds)
    log_beta_update = utils.logsumexp_nonzero(log_update, axis=1)
    return log_beta_update

def forward_recursion(localization, ref_map, query_descriptors, query_odom):
    n_q = query_descriptors.shape[0]
    odom_mu, odom_sigma = query_odom
    # sparse matrix indices for update step in fw/backward
    data_bc_inds = None
    # store measurement lhoods and transition matrices for b/w recursion
    log_meas_lhoods = np.empty((n_q, localization.belief.shape[0]))
    log_transition_probs = []
    # forward recursion joint lhood
    log_alpha = np.empty((n_q, localization.belief.shape[0]))
    for t in trange(n_q, desc='forward', leave=False):
        if t == 0:
            prior = localization.belief.copy()
            meas_lhood = np.log(localization.init(query_descriptors[t]))
            log_alpha[t, :] = np.log(prior) + meas_lhood
        else:
            motion_model = localization._update_motion(odom_mu[t], odom_sigma[t]).copy()
            motion_model.data = np.log(motion_model.data)
            if t == 1:
                data_bc_inds = utils.bc_vec_to_data_inds(motion_model.tocsr())
            meas_lhood = np.log(localization._update_meas(query_descriptors[t]))
            log_transition_probs.append(motion_model)
            log_alpha[t, :] = forward_update(log_alpha[t-1, :], motion_model, meas_lhood, data_bc_inds=data_bc_inds)
        log_meas_lhoods[t, :] = meas_lhood
    return log_alpha, log_transition_probs, log_meas_lhoods

def backward_recursion(log_trans_mats, log_meas_lhoods):
    n_q, n_r = log_meas_lhoods.shape
    # sparse matrix indices for update step in fw/backward
    data_bc_inds = utils.bc_vec_to_data_inds(log_trans_mats[0].tocsc())
    # backward recursion joint lhood
    log_beta = np.empty((n_q, n_r))
    log_beta[-1, :] = 0.
    for t in tqdm(reversed(range(n_q-1)), desc='backward', total=n_q-1, leave=False):
        log_beta[t] = backward_update(log_beta[t+1], log_trans_mats[t], log_meas_lhoods[t+1], data_bc_inds=data_bc_inds)
    return log_beta


def evaluate_proposal_error(proposals, query_gt):
    pose_ests_fw, pose_ests_bw = proposals
    n_q = query_gt.shape[0]
    assert pose_ests_fw.shape[0] == n_q
    assert pose_ests_bw.shape[0] == n_q
    gt_errs_xy_fw, gt_errs_rot_fw = utils.pose_err_elementwise(query_gt, pose_ests_fw, degrees=True)
    gt_errs_xy_bw, gt_errs_rot_bw = utils.pose_err_elementwise(query_gt, pose_ests_bw, degrees=True)
    return (gt_errs_xy_fw, gt_errs_xy_bw), (gt_errs_rot_fw, gt_errs_rot_bw)


def loop_closure_detection(localization, ref_map, query_global, odom_mu, odom_sigma):
    log_alpha, log_trans, log_meas = forward_recursion(localization, ref_map, query_global, (odom_mu, odom_sigma))
    log_beta = backward_recursion(log_trans, log_meas)
    log_marginal = logsumexp(log_alpha[-1, :])
    beliefs_forward = np.exp(log_alpha - logsumexp(log_alpha, axis=1)[:, None])
    beliefs_backward = np.exp(log_alpha + log_beta - log_marginal)
    pose_ests_fw, scores_fw = lcd_proposals(beliefs_forward, localization)
    pose_ests_bw, scores_bw = lcd_proposals(beliefs_backward, localization)
    return (scores_fw, scores_bw), (pose_ests_fw, pose_ests_bw)


def lcd_proposals(beliefs, localization):
    n_q = beliefs.shape[0]
    convergence_scores = np.empty(n_q, dtype=float)
    pose_estimates = np.empty((n_q, 3), dtype=float)
    for t in range(n_q):
        pose_est, score = localization.check_convergence(belief=beliefs[t])
        convergence_scores[t] = score
        pose_estimates[t] = pose_est
    return pose_estimates, convergence_scores
