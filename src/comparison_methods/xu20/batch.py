import numpy as np

from ours.batch import viterbi
from comparison_methods.xu20.online import OnlineLocalization


def batchlocalization(params, refMap, qOdoms, qGlbs, qLocs):
    assert len(qOdoms) == len(qGlbs) - 1
    assert len(qGlbs) == len(qLocs)

    online = OnlineLocalization(params, refMap)
    prior = online.belief.copy()

    # record input for Viterbi decoding, likelihoods and transition probs

    full_lhoods = []
    full_trans_mats = []

    # iterate forward in online manner and store beliefs
    qOdoms = np.vstack((np.zeros(3), qOdoms))  # t=0 has no odom, pad

    for t, (qOdom, qGlb, qLoc) in enumerate(zip(qOdoms, qGlbs, qLocs)):
        # retrieve beliefs at all stages of update, use to back out agg.
        if t == 0:
            full_lhood = online.init(qOdom, qGlb, qLoc)
        else:
            online._update_motion(qOdom)
            full_lhood = online._update_meas(qGlb, qLoc)
            # store full state transition matrices
            full_trans_mats.append(online.E.copy())
        # store observation likelihoods and off map detections
        full_lhoods.append(full_lhood)

    # all states constrained on/off

    prior[-1] = prior[0]
    opt_seq = viterbi(full_trans_mats, full_lhoods, prior)
    return opt_seq
