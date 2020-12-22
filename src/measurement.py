import numpy as np


def vpr_lhood(query_similarities, lhmax, lvl, alpha, k):
    w = np.array([0.6, 0.8, 1.0, 0.8, 0.6])
    w = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    #w = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    ind_max = np.argpartition(-query_similarities, k, axis=1)

    lhoods = np.zeros_like(query_similarities)
    qsims = query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]]
    qsims = (qsims - qsims.min(axis=1)[:, None])
    qsims = qsims / qsims.sum(axis=1)[:, None] * 5
    #qsims = (qsims - qsims.min(axis=1)[:, None]) / qsims.max(axis=1)[:, None]
    qsims = (qsims ** alpha) * lhmax
    qsims += lvl

    lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = qsims
    for i in range(len(lhoods)):
        lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        lhoods[i] += 1.
    # lhoods = np.exp(5.0 * query_similarities)
    return lhoods

def retrieval_fn(query_sims, k):
    # kernel fn for smoothing retrievals
    s_r = 2.
    w = np.exp(- np.arange(-5, 6) ** 2. / (2. * s_r ** 2.))

    # retrievals out
    r = np.zeros_like(query_sims)
    ind_max = np.argpartition(-query_sims, k, axis=1)
    qsims = query_sims[np.arange(len(r))[:, None], ind_max[:, :k]]
    qsims = (qsims - qsims.min(axis=1)[:, None])
    qsims = qsims / qsims.sum(axis=1)[:, None]
    # set normalized similarities as non-zero elements in retrieval fn output
    r[np.arange(len(r))[:, None], ind_max[:, :k]] = qsims
    for i in range(len(r)):
        r[i] = np.convolve(r[i], w, mode='same')
    return r


def retrieve_standardize(sims, k):
    ind_k = np.sort(np.argpartition(-sims, k, axis=1)[:, :k], axis=1)
    sims_k = sims[np.arange(len(sims))[:, None], ind_k]
    sims_norm = (sims_k - sims_k.min(axis=1)[:, None]) / \
        (sims_k.max(axis=1) - sims_k.min(axis=1))[:, None]
    return sims_norm, ind_k


def contiguous_peaks(ind_k):
    peaks = []
    offset = ind_k[:, 1:] - ind_k[:, :-1]
    close = offset <= 3.01  # retrievals are nearby, forms a peak
    for t in range(len(ind_k)):
        inds = np.diff(close[t]).nonzero()[0] + 1
        if close[t, 0]:
            inds = np.r_[0, inds]  # 0 is the first start index if segment is contiguous
        if close[t, -1]:
            inds = np.r_[inds, ind_k.shape[1]]  # k is the end index
        inds = inds.reshape(-1, 2)
        peaks.append(inds)
    return peaks


def peak_weights(peak_inds, sims_k):
    peak_wts = []
    for t in range(len(sims_k)):
        wts = [sims_k[t, peak_inds[t][j, 0]:peak_inds[t][j, 1]].sum() for j in range(len(peak_inds[t]))]
        peak_wts.append(wts)
    return peak_wts


def off_map_recursion(prior_belief_off, prior_off, sensor):
    # perform posterior update with new off-map classification
    r1 = sensor / (1. - sensor)
    r2 = (1. - prior_off) / prior_off
    r3 = (1 - prior_belief_off) / prior_belief_off
    updated_belief_off = 1. / (1. + r1 * r2 * r3)  # p(x_t = off | z_{1:t})
    return updated_belief_off


def measurement_update(prior_belief, vpr_lhoods, off_map_prob, prior_off_classif):
    new_belief = prior_belief.copy()
    new_belief[:-1] *= vpr_lhoods
    new_belief[-1] = off_map_recursion(new_belief[-1], prior_off_classif, off_map_prob)
    new_belief[:-1] *= (1. - new_belief[-1]) / new_belief[:-1].sum()
    return new_belief
