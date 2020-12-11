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


def odom_deviation(ref_map, odom, w):
    # compute deviation using vo from map
    source, des, s1, s2 = zip(*[(s, d, x["s1"], x["s2"]) for (s, d, x)
                              in ref_map.edges.data()])
    att_wt = np.ones((1, 6))
    att_wt[:, 3:] *= w
    s1 = np.asarray(s1) * att_wt
    s2 = np.asarray(s2) * att_wt
    dev = mindist_pt_seg(odom, s1, s2)
    # create deviation matrix
    N = len(ref_map) - 1
    E = sparse.csc_matrix((np.exp(-dev), (source, des)), shape=(N, N))
    dmin = -np.log(np.squeeze(E.max(axis=1).toarray()))
    dev_data = {"source": source, "des": des, "dev": dev, "dev_off": dmin}
    return dev_data


def create_transition_matrix(dev_data, params):
    """
    Creates transition matrix conditioned on odom. Takes precomputed
    odom deviation from map transitions.
    Args:
    """

    N = params["N"]

    dev = dev_data["dev"]
    dev_off = dev_data["dev_off"]
    source_indices = dev_data["source"]
    des_indices = dev_data["des"]
    theta = params["theta"]
    Eoo = params["Eoo"]
    p_min = params["p_min"]
    p_max = params["p_max"]
    d_min = params["d_min"]
    d_max = params["d_max"]
    width = params["width"]

    # probability of leaving the map from any within map node
    # (last column, excluding bottom right element)

    source_onoff = np.arange(N)
    dest_onoff = np.ones(N, dtype=np.int) * (N - 1)
    p_onoff = np.empty(N)

    p_onoff[:-1] = pw_prob(dev_off, p_min, p_max, d_min, d_max)
    p_onoff[-1] = 0.

    # adjust on-off probabilities based on number of outflow nodes
    # to prevent accumulation of mass at the end nodes

    pwts = np.ones_like(p_onoff)
    pwts[-width-1:-1] = np.linspace(1., 1. / width, width)
    p_onon = (1. - p_onoff) * pwts
    p_onoff = 1. - p_onon

    # probability of transition to other within map states given in the map
    # ((N-1) x (N-1) top left sub-block)

    E_topleft = coo_matrix((np.exp(-theta * dev),
                           (source_indices, des_indices)), shape=(N, N))
    E_topleft = normalize(E_topleft, norm='l1', axis=1)
    E_topleft = E_topleft.multiply(1. - p_onoff[:, None])

    # probability of returning to map from off-map (last row)

    source_fromoff = np.ones(N, dtype=np.int) * (N - 1)
    dest_fromoff = np.arange(N)
    p_fromoff = np.ones(N) * (1. - Eoo) / (N - 1)
    p_fromoff[-1] = Eoo

    # construct transition matrix outside of top left sub-block

    source_other = np.concatenate((source_onoff, source_fromoff))
    dest_other = np.concatenate((dest_onoff, dest_fromoff))
    p_other = np.concatenate((p_onoff, p_fromoff))
    E_other = csc_matrix((p_other, (source_other, dest_other)), shape=(N, N))

    E = E_topleft + E_other

    return E
