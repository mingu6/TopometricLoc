import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from sklearn.preprocessing import normalize


def logistic(x, theta1, theta2):
    return 1. / (1. + np.exp(- theta1 * (x - theta2)))


def shortest_dist_segments(p0, u, p1, v):
    """
    Returns shortest distance between two line segments.
    Input:
        p0: start point of first segment
        u: relative vector to endpoint of first segment
        p1: start point of second segment
        v: relative vector to endpoint of second segment
    Returns:
        dmin: minimum distance between segments
    All input/output assumes N segments (i.e. NxD), where D is
    the dimension of the space the lines lie in
    """
    w0 = p0 - p1
    a = np.linalg.norm(u, axis=-1) ** 2
    b = (u * v).sum(axis=-1)
    c = np.linalg.norm(v, axis=-1) ** 2
    d = (u * w0).sum(axis=-1)
    e = (v * w0).sum(axis=-1)

    # first find points along full, unconstrained lines which
    # correspond to the shortest distance between those lines

    denom = a * c - b ** 2
    valid = denom > 1e-10  # valid = lines not parallel
    invalid = np.squeeze(np.argwhere(np.logical_not(valid)))

    s = np.empty_like(denom)
    t = np.empty_like(denom)
    s[valid] = (b * e - c * d)[valid] / denom[valid]
    t[valid] = (a * e - b * d)[valid] / denom[valid]

    # for each edge, compute shortest distance to edge for
    # relevant edge cases

    dmin = np.ones_like(s) * np.inf

    # lines are parallel, check for collinearity first. dist = 0

    s_cl = - d[invalid] / a[invalid]  # segment b/w p0 and p1 collin w/seg 1?
    d_cl = np.linalg.norm(w0[invalid] + s_cl[..., None] * u[invalid], axis=-1)
    cl = d_cl < 1e-10  # lies on the same line
    not_cl = np.logical_not(cl)  # segments are parallel but not collinear
    not_cl = invalid[not_cl]

    # for non-collinear segments, compute parallel offset

    s[not_cl] = 0.
    t[not_cl] = np.clip(e[not_cl] / c[not_cl], 0, 1)
    dmin[not_cl] = np.linalg.norm(
        w0[not_cl] - t[not_cl, None] * v[not_cl], axis=-1)

    # for collinear, check if overlap

    s_cl_1 = b[invalid] / a[invalid] + s_cl  # dist to end of 2nd segment?
    s_cl = np.clip(s_cl, 0, 1)
    s_cl_1 = np.clip(s_cl_1, 0, 1)
    # minimum distance between segments
    d_cl_1 = np.linalg.norm(w0[invalid] - v[invalid] +
                            s_cl_1[..., None] * u[invalid], axis=-1)
    d_cl = np.linalg.norm(w0[invalid] + s_cl[..., None] * u[invalid], axis=-1)
    dmin[invalid[cl]] = np.minimum(d_cl_1, d_cl)[cl]

    # regular cases, s, t in [0, 1]

    reg = np.logical_and.reduce((s >= 0, s <= 1, t >= 0, t <= 1))
    dmin[reg] = np.linalg.norm(
        w0[reg] - t[reg, None] * v[reg] + s[reg, None] * u[reg], axis=-1)

    # if s, t outside [0, 1]^2, then check cases and find
    # constrained solution along segment

    # s = 0 edge

    s_lt_0 = s < 0
    t_min = np.clip(e[s_lt_0] / c[s_lt_0], 0, 1)
    d_t_min = np.linalg.norm(
        w0[s_lt_0] - t_min[..., None] * v[s_lt_0], axis=-1)
    dmin[s_lt_0] = np.minimum(dmin[s_lt_0], d_t_min)

    # t = 0 edge

    t_lt_0 = t < 0
    s_min = np.clip(- d[t_lt_0] / a[t_lt_0], 0, 1)
    d_s_min = np.linalg.norm(
        w0[t_lt_0] + s_min[..., None] * u[t_lt_0], axis=-1)
    dmin[t_lt_0] = np.minimum(dmin[t_lt_0], d_s_min)

    # s = 1 edge

    s_gt_1 = s > 1
    t_min = np.clip((e[s_gt_1] + b[s_gt_1]) / c[s_gt_1], 0, 1)
    d_t_min = np.linalg.norm(
        w0[s_gt_1] + u[s_gt_1] - t_min[..., None] * v[s_gt_1],
        axis=-1)
    dmin[s_gt_1] = np.minimum(dmin[s_gt_1], d_t_min)

    # t = 1 edge

    t_gt_1 = t > 1
    s_min = np.clip((b[t_gt_1] - d[t_gt_1]) / a[t_gt_1], 0, 1)
    d_s_min = np.linalg.norm(
        w0[t_gt_1] + s_min[..., None] * u[t_gt_1] - v[t_gt_1],
        axis=-1)
    dmin[t_gt_1] = np.minimum(dmin[t_gt_1], d_s_min)

    return dmin


def odom_deviation_nonself(mapG, odom, att_wt):
    source, dest, tO1, tO2, tD1, tD2, tOD = \
        zip(*[(s, d, data["tO1"], data["tO2"],
               data["tD1"], data["tD2"], data["tOD"])
              for (s, d, k, data) in mapG.edges.data(keys=True)
              if k == "nonself"])

    odomC = odom * att_wt
    tO1 = np.asarray(tO1) * att_wt
    tO2 = np.asarray(tO2) * att_wt
    tD1 = np.asarray(tD1) * att_wt
    tD2 = np.asarray(tD2) * att_wt
    tOD = np.asarray(tOD) * att_wt

    # compute best deviation
    d11 = shortest_dist_segments(tO1 + odomC, -tO1,
                                 tD1, tOD - tD1)
    d12 = shortest_dist_segments(tO1 + odomC, -tO1,
                                 tOD, tD2 - tOD)
    d21 = shortest_dist_segments(odomC, tO2,
                                 tD1, tOD - tD1)
    d22 = shortest_dist_segments(odomC, tO2,
                                 tOD, tD2 - tOD)
    d = np.stack((d11, d12, d21, d22), axis=1).min(axis=1)
    return list(source), list(dest), d


def odom_deviation_self(mapG, odom, att_wt):
    source, dest, tO1, tO2 = \
        zip(*[(s, d, data["tO1"], data["tO2"])
              for (s, d, k, data) in mapG.edges.data(keys=True)
              if k == "self"])

    odomC = odom * att_wt
    tO1 = np.asarray(tO1) * att_wt
    tO2 = np.asarray(tO2) * att_wt

    # compute best deviation
    d = shortest_dist_segments(tO1 + odomC, -tO1, np.zeros_like(tO2), tO2)
    return list(source), list(dest), d


def create_deviation_matrix(mapG, odom, Eoo, w):
    N = len(mapG)
    att_wt = np.ones(6)
    att_wt[3:] *= w

    # non-self transitions

    source_ns, dest_ns, dev_ns = odom_deviation_nonself(mapG, odom, att_wt)

    # self-transitions

    source_s, dest_s, dev_s = odom_deviation_self(mapG, odom, att_wt)

    # concatenate into single vectors

    source = np.array(source_ns + source_s, dtype=np.int)  # source node indices
    dest = np.array(dest_ns + dest_s, dtype=np.int)  # destination node indices
    dev_within = np.concatenate((dev_ns, dev_s))  # corresp. deviation values

    # lowest deviation for a given source node, used for off-map prob.

    dev_off = np.asarray([dev_within[source == node].min() for node in range(N-1)])

    return {"source": source, "dest": dest, "dev": (dev_within, dev_off)}


def create_transition_matrix(deviation_data, N, Eoo, theta1, theta2, theta3):
    """
    Creates transition matrix conditioned on odom.
    Args:
        mapG: networkx MultiDiGraph containing map transitions
        odom: R^6 odometry to set transition probabilities
        Eoo: Prob. of off-map to off-map transition
        w: attitude weight
        TODO: params
    """

    dev_within, dev_off = deviation_data["dev"]
    source_indices = deviation_data["source"]
    dest_indices = deviation_data["dest"]

    # probability of leaving the map from any within map node
    # (last column, excluding bottom right element)

    source_onoff = np.arange(N)
    dest_onoff = np.ones(N, dtype=np.int) * (N - 1)
    p_onoff = np.empty(N)
    p_onoff[:-1] = logistic(dev_off, theta1, theta2)
    p_onoff[-1] = 0.

    # probability of transition to other within map states given in the map
    # ((N-1) x (N-1) top left sub-block)

    E_topleft = coo_matrix((np.exp(-theta3 * dev_within),
                           (source_indices, dest_indices)), shape=(N, N))
    E_topleft = normalize(E_topleft, norm='l1', axis=1)
    E_topleft = E_topleft.multiply(1. - p_onoff[:, None]).tocsc()

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


def motion_off_map_prob(transition_matrix, belief):
    """
    Compute prior probability of leaving map from motion only
    Args:
        transition_matrix: State transition matrix conditioned on odometry
        belief: posterior belief from previous time period
    """
    return transition_matrix[:, -1] @ belief
