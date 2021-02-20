import time
import numpy as np

import cv2

from data.utils import preprocess_local_features

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


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


def off_map_detection(qLoc, refMap, lhoods,
                      num_feats, window, num_verif,
                      num_inliers, inlier_threshold,
                      confidence):
    qkp, qdes = preprocess_local_features(qLoc, num_feats)
    # identify peak heights and node indices in reference map to run verif. on
    verif_inds = []
    lhoods_new = lhoods.copy()
    for i in range(num_verif):
        ind_max = np.argmax(lhoods_new)
        lhoods_new[max(ind_max-window, 0): min(ind_max+window, len(lhoods))] = 0.
        verif_inds.append(ind_max)
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
    return verif_succ, success_ind


def update_off_prob(on, prior_off, p_off_off, p_on_on):
    """
    Update posterior probability of being off-map.
    Args:
        on (Boolean): Spatial verification succeeded, on-map
        prior_off (float): Prior off-map probability after motion update
        p_off_off: prob. of verif. failure given not within map
        p_on_on: prob. of verif. success given within map
    Output:
        off_new: updated off-map probability
    """
    prior_on = 1 - prior_off

    if on:
        numer = (1. - p_off_off) * prior_off
        denom = numer + p_on_on * prior_on
    else:
        numer = p_off_off * prior_off
        denom = numer + (1. - p_on_on) * prior_on
    off_new = numer / denom
    return off_new


def within_lhood(qGlb, refGlb, lambd):
    """
    Compute likelihood for within-map states using global descriptor distances.
    Args:
        qGlb (size D np array): Query image global descriptor
        refGlb (NxD np array): Reference map image descriptors
        lambd (float): calibrated likelihood parameter
    Returns:
        lhood (size N np array): observation likelihood for within-map states
    """
    query_sims = refGlb @ qGlb
    dist = np.sqrt(2. - 2. * query_sims)
    lhood = np.exp(-lambd * dist)
    return lhood


def off_map_lhood(qLoc, refMap, within_lhood, prior_belief, meas_params):
    """
    Computes implicit likelihood of off-map state. Performs geometric
    verification and uses the output to update posterior belief.
    Likelihood is computed implicitly using the previous and updated
    off-map state belief values.
    Args:
        qLoc: Local features for query image
        refMap: reference map object
        within_lhood (size N np array): within-map likelihood from global desc.
        prior_belief (size N+1 np array): belief vector after motion update
        meas_params (dict): Measurement model parameters
    Returns:
        off_lhood (float): off-map state likelihood
    """
    # extract parameters
    num_feats = meas_params['num_feats']
    window = meas_params['window']
    num_verif = meas_params['num_verif']
    num_inliers = meas_params['num_inliers']
    inlier_threshold = meas_params['inlier_threshold']
    confidence = meas_params['confidence']
    p_off_off = meas_params['p_off_off']
    p_on_on = meas_params['p_on_on']
    # detect off-map using spatial verification
    on, _ = off_map_detection(qLoc, refMap, within_lhood,
                              num_feats, window, num_verif, num_inliers,
                              inlier_threshold, confidence)
    # update off-map probability using noisy detector model
    prior_off = prior_belief[-1]
    post_off = update_off_prob(on, prior_off, p_off_off, p_on_on)
    # compute off-map implicit likelihood
    off_lhood = post_off / ((1. - post_off) * prior_off) * \
        within_lhood @ prior_belief[:-1]
    return off_lhood


def meas_update(prior_belief, lhood):
    post_belief = prior_belief * lhood
    return post_belief / post_belief.sum()
