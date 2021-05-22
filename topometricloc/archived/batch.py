import numpy as np

from scipy import sparse

from data.utils import preprocess_local_features
from ours.motion import odom_deviation, transition_probs
from ours.measurement import measurement_update, geometric_verification
from ours.online import OnlineLocalization


def viterbi(transition_mats, obs_lhoods, prior, constr=None):
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
        constr (optional): Length T+1 state constraints (int or None)
    Returns:
        opt_state_seq: Length T list of indices representing opt. state seq.
    """
    assert len(transition_mats) == len(obs_lhoods) - 1
    assert constr is None or len(constr) == len(obs_lhoods)
    assert transition_mats[0].shape[0] == transition_mats[0].shape[1]
    assert transition_mats[0].shape[0] == len(obs_lhoods[0])

    T = len(transition_mats)
    K = len(obs_lhoods[0])
    trellis = np.empty((T+1, K), dtype=np.float)
    trellis[0, :] = np.log(obs_lhoods[0]) + np.log(prior)
    ptr = np.empty((T+1, K), dtype=np.int)  # backtracing pointer

    for t in range(1, T+1):
        mat = transition_mats[t-1]
        obs_lhood = np.log(obs_lhoods[t])
        if type(mat) is np.ndarray:
            update = np.log(mat) + trellis[t-1][:, None] + obs_lhood[None, :]
            if constr is None:
                trellis[t, :] = update.max(axis=0)
                ptr[t-1, :] = np.argmax(update, axis=0)
            else:
                # constraints are either integer (single contr) or a
                # slice object (multiple options)
                sl = constr[t-1]
                trellis[t, :] = np.atleast_2d(update[sl]).max(axis=0)
                if type(sl) == int:
                    ptr[t-1, :] = sl
                else:
                    ptr[t-1, :] = sl[np.argmax(update[sl], axis=0)]
        else:
            update = np.log(np.clip(mat.toarray(), 0., 1.)) + trellis[t-1][:, None] + \
                obs_lhood[None, :]
            if constr is None:
                trellis[t, :] = update.max(axis=0)
                ptr[t-1, :] = update.argmax(axis=0)
            else:
                # constraints are either integer (single contr) or a
                # slice object (multiple options)
                sl = constr[t-1]
                trellis[t, :] = update[sl, :].max(axis=0)
                if type(sl) == int:
                    ptr[t-1, :] = sl
                else:
                    ptr[t-1, :] = update[sl].argmax(axis=0)

    # run backtracing for optimal path

    opt_seq = np.empty(T+1, dtype=np.int)
    opt_seq[-1] = np.argmax(trellis[-1, :])  # highest lhood seq.

    for t in reversed(range(T)):
        opt_seq[t] = ptr[t, opt_seq[t+1]]
    import pdb; pdb.set_trace()
    return opt_seq


def agg_trans_mat(prior_belief, pred_belief, off_off_prob):
    pred_on = 1. - pred_belief[-1]
    prior_on = 1. - prior_belief[-1]
    p_off_off = off_off_prob

    p_on_on = (pred_on - (1. - p_off_off) * (1. - prior_on)) / prior_on

    return np.array([[p_on_on, 1. - p_on_on],
                     [1. - p_off_off, p_off_off]])


def agg_lhood(lhood_off, pred_off, post_off):
    lhood_on = (1. - post_off) / post_off * pred_off / (1. - pred_off) * lhood_off
    return np.array([lhood_on, lhood_off])


def batchlocalization(params, refMap, qOdoms, qGlbs, qLocs):
    assert len(qOdoms) == len(qGlbs) - 1
    assert len(qGlbs) == len(qLocs)

    p_off_off = params["motion"]["p_off_off"]

    online = OnlineLocalization(params, refMap)
    prior = online.belief.copy()
    agg_prior = np.array([prior[:-1].sum(), prior[-1]])

    # record input for Viterbi decoding, likelihoods and transition probs

    full_lhoods = []
    agg_lhoods = []
    full_trans_mats = []
    agg_trans_mats = []

    # iterate forward in online manner and store beliefs

    for t, (qOdom, qGlb, qLoc) in enumerate(zip(qOdoms, qGlbs, qLocs)):
        # retrieve beliefs at all stages of update, use to back out agg.
        if t == 0:
            belief_pred = online.belief
            belief_post, full_lhood = online.init(qOdom, qGlb, qLoc)
        else:
            #if t in [112, 113, 114, 115, 117, 120, 135, 140, 144]:
            # if t == 150 or t == 175:
                # import pdb; pdb.set_trace()
            # if t in range(70, 90):
                # import pdb; pdb.set_trace()
            belief_prior = online.belief
            belief_pred, mat = online._update_motion(qOdom)
            belief_post, full_lhood = online._update_meas(qGlb, qLoc)
            # compute aggregate on/off transitions
            agg_trans_mats.append(agg_trans_mat(belief_prior, belief_pred,
                                                p_off_off))
            # store full state transition matrices
            full_trans_mats.append(mat)
        # compute aggregate on/off lhoods
        agg_lhoods.append(agg_lhood(full_lhood[-1], belief_pred[-1],
                                    belief_post[-1]))
        # store observation likelihoods and off map detections
        full_lhoods.append(full_lhood)

    # aggregate (on/off state only) decoding

    #agg_seq = viterbi(agg_trans_mats, agg_lhoods, agg_prior, constr=None)
    # constrain full decoding to within/off map
    #constr = [-1 if el == 1 else slice(0, -1) for el in agg_seq]

    # all states constrained on/off

    # prior[-1] = prior[0] * params["other"]["prior_off"]
    # prior[-1] = prior[0]
    #opt_seq = viterbi(full_trans_mats, full_lhoods, prior, constr=constr)
    opt_seq = viterbi(full_trans_mats, full_lhoods, prior, constr=None)
    return opt_seq
