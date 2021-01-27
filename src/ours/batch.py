import numpy as np

from scipy import sparse

from data.utils import preprocess_local_features
from ours.motion import odom_deviation, transition_probs
from ours.measurement import measurement_update, geometric_verification
from ours.online import OnlineLocalization


def viterbi(transition_mats, obs_lhoods, prior):
    """
    Runs the Viterbi decoding algorithm given state transition matrices and
    observation likelihoods (each length T). Also allows the application of
    optional constraints of a length T vector e.g. [2, None, 3, 1, None]
    which provide state constraints (i.e. at time t the state is locked
    at state number constr[t], where None means no constraint).
    Args:
        transition_mats: Length T list of KxK state transition matrices
        obs_lhoods: Length T+1 list of Kx1 observation lhoods for states
        prior: prior Nx1 state probabilities at t=0
    Returns:
        opt_state_seq: Length T list of indices representing opt. state seq.
    """
    assert len(transition_mats) == len(obs_lhoods) - 1
    assert transition_mats[0].shape[0] == transition_mats[0].shape[1]
    assert transition_mats[0].shape[0] == len(obs_lhoods[0])

    T = len(transition_mats)
    K = len(obs_lhoods[0])
    trellis = np.empty((T+1, K), dtype=np.float)
    trellis[0, :] = np.log(obs_lhoods[0]) + np.log(prior)
    ptr = np.empty((T+1, K), dtype=np.int)  # backtracing pointer

    for t in range(1, T+1):
        # evaluate all operations for Viterbi in log-space
        tmat = transition_mats[t-1]
        tmat.data = np.log(tmat.data)
        # for ons lhood and trellis term in updates, create new sparse matrix
        # to allow sum updates in log-space
        row_nz, col_nz = tmat.nonzero()
        lhood_mat = sparse.csc_matrix((np.log(obs_lhoods[t])[col_nz],
                                       (row_nz, col_nz)), shape=tmat.shape)
        trellis_mat = sparse.csc_matrix((trellis[t-1][row_nz], (row_nz, col_nz)),
                                        shape=tmat.shape)
        update = tmat + lhood_mat + trellis_mat
        # transform update so taking min of transformed yields max of untransform
        # transform because taking argmax of update will yield 0
        update.data = 1. / update.data  # higher value becomes lower
        trellis[t, :] = np.squeeze(1. / update.min(axis=0).toarray())  # transform back
        ptr[t-1, :] = np.squeeze(np.asarray(update.argmin(axis=0)))

    # run backtracing for optimal path

    opt_seq = np.empty(T+1, dtype=np.int)
    opt_seq[-1] = np.argmax(trellis[-1, :])  # highest lhood seq.

    for t in reversed(range(T)):
        opt_seq[t] = ptr[t, opt_seq[t+1]]
    return opt_seq


def batchlocalization(params, refMap, qOdoms, qGlbs, qLocs):
    assert len(qOdoms) == len(qGlbs) - 1
    assert len(qGlbs) == len(qLocs)

    online = OnlineLocalization(params, refMap)
    prior = online.belief.copy()

    # record input for Viterbi decoding, likelihoods and transition probs

    lhoods = []
    trans_mats = []

    # iterate forward in online manner and store beliefs
    qOdoms = np.vstack((np.zeros(3), qOdoms))  # t=0 has no odom, pad

    for t, (qOdom, qGlb, qLoc) in enumerate(zip(qOdoms, qGlbs, qLocs)):
        # retrieve beliefs at all stages of update, use to back out agg.
        if t == 0:
            lhood = online.init(qOdom, qGlb, qLoc)
        else:
            mat = online._update_motion(qOdom)
            lhood = online._update_meas(qGlb, qLoc)
            # store full state transition matrices
            trans_mats.append(mat)
        # store observation likelihoods (off-map lhood computed implicitly)
        lhoods.append(lhood)

    opt_seq = viterbi(trans_mats, lhoods, prior)
    return opt_seq
