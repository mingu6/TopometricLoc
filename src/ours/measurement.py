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


def measurement_update(prior_belief, query_sims, qLoc, refMap, mment_params,
                       lambd, noverif=False, nooff=False):
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
        likelihood (N + 1) including implicit off-map likelihood
    """
    dist = np.sqrt(2. - 2. * query_sims)
    vpr_lhood = np.exp(-lambd * dist)
    # performs geometric verification for top few peaks in retrievals
    if not noverif or nooff:
        on_detected, _ = off_map_detection(
            qLoc, refMap, vpr_lhood, mment_params['num_feats'],
            mment_params['window'], mment_params['num_verif'],
            mment_params['num_inliers'], mment_params['inlier_threshold'],
            mment_params['confidence']
        )
    if noverif:
        p_off_off = 0.5
        p_on_on = 0.5
        on_detected = False
    else:
        p_off_off = mment_params['p_off_off']
        p_on_on = mment_params['p_on_on']
    # off-map prob. update step
    if not nooff:
        off_new = update_off_prob(not on_detected, prior_belief,
                                  p_off_off, p_on_on)
        # compute implicit likelihood for off-map state
        off_lhood = off_new / ((1. - off_new) * prior_belief[-1]) * \
            vpr_lhood @ prior_belief[:-1]
        lhoods = np.concatenate((vpr_lhood, [off_lhood]))
    else:
        lhoods = vpr_lhood
    # update belief using lhood
    updated_belief = lhoods * prior_belief
    updated_belief /= updated_belief.sum()
    return updated_belief, lhoods
