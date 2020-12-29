import time
import numpy as np

import cv2

from data.utils import preprocess_local_features

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


def retrieval_fn(query_sims, k, window, s_r, rho, alpha):
    """
    Implements r(x) in the paper, takes top-k retrievals and smooths them
    spatially using a kernel with window (nhood) size window and bandwidth
    s_r. Scales using rho and applies power law normalization using alpha.
    """
    # kernel fn for smoothing retrievals
    w = np.exp(- np.arange(-window, window+1) ** 2. / (2. * s_r ** 2.))
    # retrievals out
    r = np.zeros_like(query_sims)
    ind_max = np.argpartition(-query_sims, k)
    qsims = query_sims[ind_max[:k]]
    qsims = ((qsims - qsims.min()) / (qsims.max() - qsims.min())) ** alpha
    qsims = rho * (qsims / qsims.sum())
    # set normalized similarities as non-zero elements in retrieval fn output
    r[ind_max[:k]] = qsims
    r = np.convolve(r, w, mode='same')
    return r


def contiguous_peaks(retrievals):
    """
    Given retrieval fn output, identifies "peaks", where each peak is defined as
    a contiguous section of non-zero values. Returns peak indices as an Nx2
    matrix where column 1 is the start index and column 2 is the end index.
    """
    ind_gt_0 = retrievals > 0.
    inds = np.diff(ind_gt_0).nonzero()[0]
    if ind_gt_0[0]:
        inds = np.r_[0, inds]  # 0 is the first start index if segment is contiguous
    if ind_gt_0[-1]:
        inds = np.r_[inds, len(retrievals)]  # k is the end index
    inds = inds.reshape(-1, 2)
    return np.atleast_2d(inds)


def peak_heights(retrievals, peak_inds):
    """
    Given retrieval values and peak indices, return the maximum height of each
    peak.
    """
    heights = []
    for inds in peak_inds:
        ind_max = np.arange(inds[0], inds[1])[np.argmax(retrievals[inds[0]:inds[1]])]
        heights.append((ind_max, retrievals[ind_max]))
    return heights


def geometric_verification(kp1, des1, kp2, des2,
                           num_inliers, inlier_threshold, confidence):
    """
    Given keypoints and descriptors from two images, check if a possible
    3D transform can be fitted, where success is defined by a threshold on the
    number of inliers.
    """
    matches = bf.match(des1, des2)  # brute force matching, filter using mutual test
    # save filtered set of 2d keypoints after matching
    pts1 = []
    pts2 = []
    for i,m in enumerate(matches):
        if m:
            pts2.append(kp2[m.trainIdx])
            pts1.append(kp1[m.queryIdx])
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # use points to find fundamental matrix F
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                     inlier_threshold, confidence)
    return mask.sum() >= num_inliers


def off_map_detection(qLoc, refMap, retrievals,
                      num_feats, num_verif, verif_multiplier,
                      num_inliers, inlier_threshold,
                      confidence):
    qkp, qdes = preprocess_local_features(qLoc, num_feats)
    # identify peak heights and node indices in reference map to run verif. on
    peak_inds = contiguous_peaks(retrievals)
    # enumerate below is to return index with sorted list
    heights = sorted(enumerate(peak_heights(retrievals, peak_inds)),
                     key=lambda x: -x[1][1])[:num_verif]
    verif_inds = [h[1][0] for h in heights]
    temp_inds = np.asarray([h[0] for h in heights])  # indices of peak arr.
    # load reference local features for verification from disk
    refLoc = [refMap.load_local(ind, num_feats) for ind in verif_inds]
    # run verification step between query and reference(s)
    verif = [geometric_verification(kp, des, qkp, qdes,
                                    num_inliers, inlier_threshold,
                                    confidence) for kp, des in refLoc]
    verif_succ = any(verif)
    # find node index where verification was a success if any
    try:
        success_ind = verif_inds[next(i for i, v in enumerate(verif) if v)]
    except StopIteration:
        success_ind = None
    # for successful verification, increase lhood of peak in retrievals
    for inds in peak_inds[temp_inds[verif], :]:
        retrievals[inds[0]:inds[1]] *= verif_multiplier
    return retrievals, verif_succ, success_ind


def update_off_prob(off_map_sensor, prior_belief, p_off_off, p_on_on):
    """
    Update posterior probability of being off-map.
    Args:
        off_map_sensor (Boolean): Sensor (geom. verif.) detecting off-map-ness
        prior_belief (N+1 np array): Belief over state after motion prediction step
        p_off_off: prob. of observing off-map from sensor given true state is off-map
        p_on_on: prob. of observing within-map from sensor given true state is within-map
    Output:
        off_map_prob: updated off-map probability
    """
    prior_belief_off = prior_belief[-1]
    prior_belief_on = 1 - prior_belief_off

    if off_map_sensor:
        numer = p_off_off * prior_belief_off
        denom = numer + (1. - p_on_on) * prior_belief_on
    else:
        numer = (1. - p_off_off) * prior_belief_off
        denom = numer + p_on_on * prior_belief_on
    return numer / denom


def update_within_map_probs(prior_belief, updated_off_prob, retrievals):
    """
    Update posterior probability of within-map states. Requires update of
    off-map state probability first.
    Args:
        prior_belief (N+1 np array): Belief over state after motion prediction step
        updated_off_prob (float): Probability of off-map state after measurement update
        retrievals (N np array): Retrieval function output utilizing vpr retrievals
    Returns:
        updated within map belief (N np array)
    """
    g = 1. + retrievals  # obs. l'hood up to propn. const.
    updated_lhood = g * prior_belief[:-1]
    return updated_lhood / updated_lhood.sum() * (1. - updated_off_prob)


def measurement_update(prior_belief, query_sims, qLoc, refMap, mment_params):
    """
    Applies measurement update for belief after motion prediction step.
    Returns new posterior belief after measurement update.
    Args:
        prior_belief (N+1 np array): Belief over state after motion prediction step
        query_sims (N np array): Image retrieval similarities from NetVLAD
        qLoc (dict): contains query image local features object
        refMap (RefMap): Reference map
        mment_params (dict): Relevant parameters for measurement model
    Returns:
        updated posterior belief (N + 1 np array)
    """
    # retrieval fn for within-map update and geom. verif.
    r = retrieval_fn(query_sims, mment_params['k'],
                     mment_params['smoothing_window'],
                     mment_params['smoothing_bandwidth'],
                     mment_params['rho'], mment_params['alpha'])
    # performs geometric verification for top few peaks in retrievals
    retrievals, on_detected, _ = off_map_detection(
        qLoc, refMap, r, mment_params['num_feats'], mment_params['num_verif'],
        mment_params['verif_multiplier'], mment_params['num_inliers'],
        mment_params['inlier_threshold'], mment_params['confidence']
    )
    # off-map prob. update step
    off_new = update_off_prob(not on_detected, prior_belief,
                              mment_params['p_off_off'], mment_params['p_on_on'])
    # within-map prob. update step
    within_new = update_within_map_probs(prior_belief, off_new, r)
    return np.hstack((within_new, off_new))
