import numpy as np


def vpr_lhood(query_similarities, lhmax, lvl, alpha, k):
    w = np.array([0.6, 0.8, 1.0, 0.8, 0.6])
    ind_max = np.argpartition(-query_similarities, k, axis=1)

    lhoods = np.zeros_like(query_similarities)
    qsims = query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]]
    qsims = (qsims - qsims.min(axis=1)[:, None]) / qsims.max(axis=1)[:, None]
    qsims = (qsims ** alpha) * lhmax
    qsims += lvl

    lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = qsims
    for i in range(len(lhoods)):
        lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        lhoods[i] += 1.
    # lhoods = np.exp(5.0 * query_similarities)
    return lhoods


def retrieve_standardize(sims, k):
    ind_k = np.sort(np.argpartition(-sims, k, axis=1)[:, :k], axis=1)
    sims_k = sims[np.arange(len(sims))[:, None], ind_k]
    sims_norm = (sims_k - sims_k.min(axis=1)[:, None]) / \
        (sims_k.max(axis=1) - sims_k.min(axis=1))[:, None]
    return sims_norm, ind_k


def contiguous_peaks(ind_k):
    peaks = []
    offset = ind_k[:, 1:] - ind_k[:, :-1]
    close = offset < 3.  # retrievals are nearby, forms a peak
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


def off_map_features(sims, k):
    sims_k, ind_k = retrieve_standardize(sims, k)
    peaks = contiguous_peaks(ind_k)
    peak_wts = peak_weights(peaks, sims_k)

    # features
    max_mass = [round(max(wts) / sk.sum(), 2) for wts, sk in zip(peak_wts, sims_k)]
    num_peaks = [round(1. - (p[:, 1] - p[:, 0]).sum() / k, 2) for p in peaks]
    peak_ratio = [round(sorted(wts)[-min(len(wts), 2)] / max(wts) , 2) if len(wts) > 1 else 0.
                  for wts, sk in zip(peak_wts, sims_k)]
    features = np.vstack((max_mass, num_peaks, peak_ratio)).T
    return features


def scaled_logistic(logits, pmin, pmax):
    return 1. / (1. + np.exp(-logits)) * (pmax - pmin) + pmin


def off_map_prob(features, pmin, pmax):
    coef = np.array([-1.5, 0.6, -1.0])
    logit = features @ coef
    return scaled_logistic(logit, pmin, pmax)


def vmflhood(query_similarities, kappas):
    k = 100
    w = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6])
    w = np.array([0.6, 0.8, 1.0, 0.8, 0.6])
    ind_max = np.argpartition(-query_similarities, k, axis=1)
    #ind_max = np.argsort(-query_similarities, axis=1)[:, :k]
    #lhoods = np.zeros_like(query_similarities)
    #lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = 0.2
    #lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = np.linspace(0.3, 0.05, k)[None, :]
    #lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = 1.0
    #for i in range(len(lhoods)):
        #lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        #lhoods[i] += 1.
    #lhoods = np.exp(8.0 * query_similarities)
    lhoods = np.zeros_like(query_similarities)
    #qsims = np.exp(kappas * query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]])
    qsims = query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]]
    qsims -= qsims.min(axis=1)[:, None]
    qsims /= qsims.max(axis=1)[:, None]
    qsims = qsims ** 0.8
    qsims *= 0.5

    #qsims *= 0.5
    #qdiff = qsims.max(axis=1) - qsims.min(axis=1)
    #qsims = (qsims - qsims.min(axis=1)[:, None]) / qdiff[:, None]
    #qsims *= 0.5
    #qsims += 0.05
    lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = qsims
    for i in range(len(lhoods)):
        lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        lhoods[i] += 1.

    return lhoods


def vmflhood(query_similarities, kappas):
    k = 100
    w = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6])
    w = np.array([0.6, 0.8, 1.0, 0.8, 0.6])
    ind_max = np.argpartition(-query_similarities, k, axis=1)
    #ind_max = np.argsort(-query_similarities, axis=1)[:, :k]
    #lhoods = np.zeros_like(query_similarities)
    #lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = 0.2
    #lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = np.linspace(0.3, 0.05, k)[None, :]
    #lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = 1.0
    #for i in range(len(lhoods)):
        #lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        #lhoods[i] += 1.
    #lhoods = np.exp(8.0 * query_similarities)
    lhoods = np.zeros_like(query_similarities)
    #qsims = np.exp(kappas * query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]])
    qsims = query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]]
    qsims -= qsims.min(axis=1)[:, None]
    qsims /= qsims.max(axis=1)[:, None]
    qsims = qsims ** 0.8
    qsims *= 0.5

    #qsims *= 0.5
    #qdiff = qsims.max(axis=1) - qsims.min(axis=1)
    #qsims = (qsims - qsims.min(axis=1)[:, None]) / qdiff[:, None]
    #qsims *= 0.5
    #qsims += 0.05
    lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = qsims
    for i in range(len(lhoods)):
        lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        lhoods[i] += 1.

    return lhoods


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
