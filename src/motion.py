import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csc_matrix, coo_matrix
from sklearn.preprocessing import normalize


def pw_prob(d, p_min, p_max, d_min, d_max):
    off_prob = np.empty_like(d)
    off_prob[d <= d_min] = p_min
    lin_mask = np.logical_and(d >= d_min, d <= d_max)
    off_prob[lin_mask] = p_min + (d[lin_mask] - d_min) * (p_max - p_min) / (d_max - d_min)
    off_prob[d > d_max] = p_max
    return off_prob


def mindist_pt_seg(p, s1, s2):
    """
    Returns the shortest distance between a point and a line segment in R^d.
    Input:
        p: point (Nxd)
        s1: starting point of segment (Nxd)
        s2: endpoint of segment (Nxd)
    Output:
        dist: minimum distances between point and line segment (Nx1)
    """
    assert s1.ndim == 2
    assert s1.shape == s2.shape
    assert len(p) == s1.shape[1]
    seg = s2 - s1
    # find optimal location in line that yields minimum distance
    t = ((p[None, :] - s1) * seg).sum(axis=1)
    t[t != 0.] /= np.linalg.norm(seg[t != 0.], axis=1)
    # project to segment
    np.clip(t, 0., 1., out=t)
    dist = np.linalg.norm(t[:, None] * seg + s1 - p[None, :], axis=1)
    return dist


def odom_deviation(qOdom, refMap, att_wt):
    N = refMap.N
    wd = refMap.width
    # extract and reshape start/end segments for odom comparison
    start_segment = refMap.odom_segments[..., 0, :].reshape(-1, 3)
    end_segment = refMap.odom_segments[..., 1, :].reshape(-1, 3)
    wt_vec = np.array([1., 1., att_wt])  # scale orientation
    # compute odom deviations using segments
    devs = mindist_pt_seg(qOdom * wt_vec, start_segment, end_segment)
    return devs.reshape(N, wd+1)


def transition_probs(deviations, p_min, p_max, d_min, d_max, theta):
    """
    Creates transition probabilities for prediction step given odometric
    deviations computed from "odom_deviation" function.
    """
    N, wd = deviations.shape

    # compute off-map transition probabilities

    min_dev = deviations.min(axis=1)  # least deviation used for off-map prob.
    off_prob = pw_prob(min_dev, p_min, p_max, d_min, d_max)
    # reweight off-map probabilities for final few nodes since they have
    # a fewer number of successor nodes, making it more likely to accumulate
    # probability mass during localization. Weighting involves decreasing
    # the total on-map probability proportional to the deficiency in
    # number of successors relative to width
    off_wts = np.linspace(1., 1. / wd, wd)
    off_prob[-wd-1:-1] = 1. - (1. - off_prob[-wd-1:-1]) * off_wts

    # compute within-map probabilities, rows are source nodes
    within_prob = np.exp(-theta * deviations)
    # zero out bottom rows corresponding to source nodes near end, no outflow
    # of probability beyond end node
    within_prob[-wd:, :] = np.fliplr(np.triu(np.fliplr(within_prob[-wd:, :])))
    within_prob /= within_prob.sum(axis=1)[:, None]
    within_prob *= (1. - off_prob)[:, None]

    return within_prob, off_prob
