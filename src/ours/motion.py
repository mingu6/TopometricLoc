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


def capped_logistic(x, theta1, theta2, ymin, ymax):
    return ymin + (ymax - ymin) * 1. / (1. + np.exp(-theta1 * (x - theta2)))


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
    t[t != 0.] /= np.linalg.norm(seg[t != 0.], axis=1) ** 2
    # project to segment
    np.clip(t, 0., 1., out=t)
    dist = np.linalg.norm(t[:, None] * seg + s1 - p[None, :], axis=1)
    return dist


def odom_deviation(qOdom, odom_segments, att_wt):
    N = len(odom_segments)
    wd = odom_segments.shape[1]
    # extract and reshape start/end segments for odom comparison
    start_segment = odom_segments[..., 0, :].reshape(-1, 3)
    end_segment = odom_segments[..., 1, :].reshape(-1, 3)
    wt_vec = np.array([1., 1., att_wt])  # scale orientation
    # compute odom deviations using segments
    devs = mindist_pt_seg(qOdom * wt_vec, start_segment * wt_vec[None, :],
                          end_segment * wt_vec[None, :])
    return devs.reshape(N, wd)


def transition_probs(deviations, p_min, p_max, d_min, d_max, theta):
    """
    Creates transition probabilities for prediction step given odometric
    deviations computed from "odom_deviation" function.
    """
    N, wd = deviations.shape

    # compute off-map transition probabilities

    min_dev = deviations.min(axis=1)  # least deviation used for off-map prob.
    off_prob = pw_prob(min_dev, p_min, p_max, d_min, d_max)

    # compute within-map probabilities, rows are source nodes
    within_prob = np.exp(-theta * deviations)
    within_prob /= within_prob.sum(axis=1)[:, None]
    within_prob *= (1. - off_prob)[:, None]

    return within_prob, off_prob
