import numpy as np

from motion_model import motion_off_map_prob


def vpr_lhood(query_similarities, lhmax, alpha, k):
    w = np.array([0.6, 0.8, 1.0, 0.8, 0.6])
    ind_max = np.argpartition(-query_similarities, k, axis=1)

    alpha = 0.7
    lhmax = 1.0
    lvl = 0.2
    k = 30

    lhoods = np.zeros_like(query_similarities)
    qsims = query_similarities[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]]
    qsims = (qsims - qsims.min(axis=1)[:, None]) / qsims.max(axis=1)[:, None]
    qsims = (qsims ** alpha) * lhmax
    qsims += lvl

    lhoods[np.arange(lhoods.shape[0])[:, None], ind_max[:, :k]] = qsims
    for i in range(len(lhoods)):
        lhoods[i] = np.convolve(lhoods[i], w, mode='same')
        lhoods[i] += 1.

    lhoods = np.exp(5.0 * query_similarities)
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


def laplacelhood(query_dists, kappas):
    """
    TO DO: product of independent laplace for SAD and BoW
    """
