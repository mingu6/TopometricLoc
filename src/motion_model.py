import numpy as np


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
