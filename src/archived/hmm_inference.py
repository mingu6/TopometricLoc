from functools import reduce
import numpy as np
from tqdm import tqdm

from scipy.optimize import minimize, Bounds
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize

from motion_model import shortest_dist_segments, create_transition_matrix, create_deviation_matrix
from measurement_model import vmflhood, measurement_update


def forward_recursion(vpr_lhood, transition_matrix, off_map_prob, alpha_prev,
                      prior_off_classif, initial=False):
    """
    Compute joint likelihood of current state and all previous
    (including current) observations using the forward recursion.
    Concretely, this function yields p(x_t, z_{1:t}| u_{1:t}).

    NOTE: If initial=True, alpha_prev must be prior belief at time 0
    """
    # compute factor used to rescale inverse sensor (off-map) measurements

    # compute prior belief after motion update
    if not initial:
        fw_lhood_prior = transition_matrix.T @ alpha_prev
    else:
        fw_lhood_prior = alpha_prev
    prior_belief = fw_lhood_prior / fw_lhood_prior.sum()  # p(x_t = off | z_{1:t-1})
    # perform posterior update with new off-map classification
    r1 = prior_off_classif / (1. - prior_off_classif)
    r2 = (1. - off_map_prob) / off_map_prob
    r3 = (1 - prior_belief[-1]) / prior_belief[-1]
    updated_belief_off = 1. / (1. + r1 * r2 * r3)  # p(x_t = off | z_{1:t})
    # compute scale factor for new forward lhood
    scale_factor = prior_belief[:-1] @ vpr_lhood / (1. - updated_belief_off)

    # import matplotlib.pyplot as plt
    # print(f"off update: {updated_belief_off} sensor: {off_map_prob} prior {prior_off_classif} prev: {prior_belief[-1]}")
    # plt.plot(alpha_prev[:-1] / alpha_prev.sum()); plt.title(f"prior {alpha_prev[:-1].sum() / alpha_prev.sum()}");
    # plt.plot(transition_matrix[:-1, -1].toarray()[:, 0], color="purple")
    # plt.show()
    # plt.plot(prior_belief[:-1]); plt.title(f"prior motion {prior_belief[:-1].sum()}"); plt.show()
    # plt.plot(vpr_lhood)
    # plt.scatter(np.argpartition(-vpr_lhood, 20)[:20], vpr_lhood[np.argpartition(-vpr_lhood, 20)[:20]])
    # plt.title("lhood"); plt.show()

    # import pdb; pdb.set_trace()

    # compute recursion

    lhood_off = updated_belief_off * scale_factor / prior_belief[-1]
    lhood_on = (1. - updated_belief_off) * scale_factor / (1. - prior_belief[-1])
    lhoods = np.append(vpr_lhood, lhood_off)
    fw_lhood = fw_lhood_prior * lhoods
    # import pdb; pdb.set_trace()

    return fw_lhood, lhood_off, lhood_on


def forward_algorithm(nv_lhoods, transition_matrices, prior_off_classif,
                      off_map_prob, prior):

    T = nv_lhoods.shape[0]
    N = len(prior)
    out = np.empty((T, N))
    off_map_lhoods = np.empty(T)  # obs lhood of off-map p(z_t | x_t = off)
    on_map_lhoods = np.empty(T)  # obs lhood of within-map p(z_t | x_t ne off)

    for t in range(T):
        if t == 0:
            out[t], off_map_lhoods[t], on_map_lhoods[t] = forward_recursion(
                nv_lhoods[t], transition_matrices[t-1],
                off_map_prob[t], prior, prior_off_classif, initial=True
            )
        else:
            out[t], off_map_lhoods[t], on_map_lhoods[t] = forward_recursion(
                nv_lhoods[t], transition_matrices[t-1], off_map_prob[t],
                out[t-1], prior_off_classif, initial=False
            )
    return out, off_map_lhoods, on_map_lhoods


def backward_recursion(vpr_lhood, transition_matrix, beta_prev, off_map_lhoods):
    lhoods = np.append(vpr_lhood, off_map_lhoods)
    bw_lhood = transition_matrix @ (lhoods * beta_prev)

    return bw_lhood


def backward_algorithm(nv_lhoods, transition_matrices, prior,
                       off_map_lhoods):

    T = nv_lhoods.shape[0]
    N = len(prior)
    out = np.empty((T, N))
    out[-1, :] = 1.  # initial conditions for recursion

    for t in reversed(range(T-1)):
        out[t] = backward_recursion(nv_lhoods[t+1], transition_matrices[t],
                                    out[t+1], off_map_lhoods[t+1])

    return out


def cross_time_prob(vpr_lhoods, off_map_lhoods, transition_matrices,
                    alpha, beta, total_lhood):
    lhoods = np.concatenate((vpr_lhoods, off_map_lhoods[:, None]), axis=1)

    # partially compute cross term (all except transition prob)

    xi = [E.multiply(a[:, None] / total_lhood).multiply((b * l)[None, :]).tocsc() for
          E, a, b, l in zip(transition_matrices, alpha[:-1], beta[1:], lhoods[1:])]

    return xi


def forward_backward(sims, deviations, off_map_prob, prior,
                     prior_off_classif, theta, kappa, Eoo, lambda1):

    T = len(sims)
    N = len(prior)

    # preprocess measurement and motion models

    transition_matrices = [create_transition_matrix(deviations[t], N, Eoo,
                                                    theta[t, 0], theta[t, 1],
                                                    theta[t, 2])
                           for t in range(T-1)]
    nv_within_lhoods = vmflhood(sims, kappa)

    # forward recursion
    alpha, off_map_lhoods, on_map_lhoods = forward_algorithm(nv_within_lhoods, transition_matrices,
                                              prior_off_classif, off_map_prob, prior)
    total_lhood = alpha[-1].sum()  # marginal lhood of obs

    # backward recursion

    beta = backward_algorithm(nv_within_lhoods, transition_matrices,
                              prior, off_map_lhoods)

    # marginal posterior belief

    gamma = alpha * beta / total_lhood

    # expected no. of transitions (cross density)

    xi = cross_time_prob(nv_within_lhoods, off_map_lhoods, transition_matrices,
                         alpha, beta, total_lhood)

    return gamma, xi


def compute_objective(params, sims, deviations, off_map_probs, prior,
                      gamma, xi):

    T = len(sims)
    N = len(prior)

    # unpack params

    theta = params[:T]
    kappa = params[T:2*T+1]
    Eoo = params[2*T+1]
    lambda1 = params[-1]

    # preprocess measurement and motion models

    transition_matrices = [create_transition_matrix(deviations[t], N, Eoo,
                                                    theta[t, 0], theta[t, 1],
                                                    theta[t, 2])
                           for t in range(T-1)]
    for E in transition_matrices:
        E.data = np.log(E.data)  # compute log probabilities for E step
    nv_within_lhoods = vmflhood(sims, kappa)

    # full forward recursion is required to compute observation lhoods
    # for off-map states

    _, off_map_lhoods = forward_algorithm(nv_within_lhoods, transition_matrices,
                                          off_map_probs, prior)
    lhoods = np.concatenate((nv_within_lhoods, off_map_lhoods[:, None]), axis=1)

    # compute objective

    obj_prior = prior @ gamma[0]
    obj_motion = [E.multiply(x).sum() for E, x in zip(transition_matrices, xi)]
    obj_meas = (np.log(lhoods) * gamma).sum()

    # regularization

    return obj_prior + obj_motion + obj_meas


def viterbi(obs_lhoods, transition_matrices, prior):
    T = obs_lhoods.shape[0]

    V = np.zeros_like(obs_lhoods)
    V[0, :] = prior * obs_lhoods[0]
    ptr = np.empty((T - 1, obs_lhoods.shape[1]), dtype=np.int)

    # run main step, compute path lhood and save backtracing ptrs

    for t in range(1, T):
        if type(transition_matrices[t-1]) is np.ndarray:
            update = transition_matrices[t-1] * V[t-1][:, None] * obs_lhoods[t][None, :]
            V[t, :] = update.max(axis=0)
            ptr[t-1] = np.argmax(update, axis=0)
        else:
            update = transition_matrices[t-1].multiply(V[t-1][:, None])\
                        .multiply(obs_lhoods[t][:, None])
            V[t, :] = update.tocsc()[:-2, :].max(axis=0).toarray()[0, :]
            ptr[t-1] = np.array(update.tocsc()[:-2, :].argmax(axis=0))[0, :]

    # run backtracing for optimal path

    opt_seq = np.empty(T, dtype=np.int)
    opt_seq[-1] = np.argmax(V[-1, :-2])

    for t in reversed(range(T-1)):
        opt_seq[t] = ptr[t, opt_seq[t+1]]

    return opt_seq

def highlevel_viterbi(on_map_lhoods, off_map_lhoods, transition_matrices, prior):

    # create aggregated transition matrices

    agg_transition_matrices = []
    for mat in transition_matrices:
        # aggregate transition matrix
        agg_mat = np.empty((2, 2))
        agg_mat[0, 1] = mat[:-1, -1]
        agg_mat[0, 0] = 1. - agg_mat[0, 1]
        agg_mat[1, 0] = 1. - mat[-1, -1]
        agg_mat[1, 1] = mat[-1, -1]
        agg_transition_matrices.append(agg_mat)

    # create lhood vectors
    lhood = np.vstack((on_map_lhoods, off_map_lhoods)).T

    # aggregate prior
    agg_prior = np.empty(2)
    agg_prior[0] = 1. - prior[-1]
    agg_prior[1] = prior[-1]

    # run viterbi decoding to find on-off sequence
    opt_seq = viterbi(lhood, agg_transition_matrices,)
    import pdb; pdb.set_trace()


def online_localization(vpr_lhoods, deviations, off_map_probs, prior,
                        prior_off_classif, Eoo, theta,
                        Tmax=100, window=5, mass_thres=0.1, ratio=0.3):

    T = vpr_lhoods.shape[0]
    N = len(prior)

    # t = 0, initial belief before motion
    belief = prior.copy()

    # convergence detection filter
    fil = np.ones(2 * window + 1)

    for t in range(T):
        if t != 0:
            transition_matrix = create_transition_matrix(
                deviations[t-1],  N, Eoo, theta[t, 0], theta[t, 1], theta[t, 2])
            belief = transition_matrix.T @ belief
        belief = measurement_update(belief, vpr_lhoods,
                                    prior_off_classif, off_map_probs[0])

        # check if belief has converged and localize if so

        # case 1: off map

        if belief[-1] > 0.95:
            return "off"

        # case 2: within map, concentration of mass
        if belief[-1] < 0.05:
            # identify two largest peaks in belief
            peak_masses = np.convolve(belief[:-1], fil, mode='same')  # extend for loop closures
            max_peaks_ind = np.argpartition(peak_masses, 2)[:2]
            max_peaks_ind = max_peaks_ind[np.argsort(peak_masses[max_peaks_ind])]
            # check threshold and ratio
            if peak_masses[max_peaks_ind[0]] / peak_masses[max_peaks_ind[1]] > ratio\
                    and peak_masses[max_peaks_ind[0]] > mass_thres:
                return max_peaks_ind[0]

    return "failure"  # failed to localize at all


def hierarchical_viterbi():
    asdas


def compute_objective1(gamma, xi, prior, sims, deviations,
                       theta, kappa, Eoo, lambda1):
    T, N = gamma.shape

    nv_llhood = kappa[:, None] * sims

    obj = (gamma[0] * np.log(prior)).sum() + \
        ((nv_llhood - lambda1 / 2. * kappa[:, None] ** 2) * gamma).sum()

    for t in range(T-1):
        theta1, theta2, theta3 = theta[t, 0], theta[t, 1], theta[t, 2]
        mat = create_transition_matrix(deviations[t], N, Eoo,
                                       theta1, theta2, theta3)
        mat.data = np.log(mat.data)
        obj += (mat.multiply(xi[t])).sum()

    return obj


def l_theta12(theta12, xi, opt_dev):
    """
    Component of the marginal log-likelihood to do with off map probability
    classifier. Used to optimize for theta_1 and theta_2 in the M-step. Note
    that this is for a SINGLE TIME STEP.
    """

    theta1 = theta12[0]
    theta2 = theta12[1]

    # within map transitions
    expterm = -theta1 * (opt_dev - theta2)
    logprob = expterm - np.log(1. + np.exp(expterm))
    tot = xi[:-1, :-1].multiply(logprob[:, None]).sum()

    # off map transitions
    tot += xi[:-1, -1].toarray()[:, 0].dot(logprob - expterm)

    return -tot

def l_theta12_jac(theta12, xi, opt_dev):
    """
    Jacobian of oomponent of the marginal log-likelihood to do with off map
    probability classifier. Used to optimize for theta_1 and theta_2 in the
    M-step. Note that this is for a SINGLE TIME STEP.
    """
    theta1 = theta12[0]
    theta2 = theta12[1]

    # within map transitions

    tot1 = 0.
    prob_off = 1. / (1. + np.exp(-theta1 * (opt_dev - theta2)))
    tot1 -= xi[:-1, :-1].multiply((opt_dev[:, None] - theta2) *
             prob_off[:, None]).sum()  # theta_1
    tot2 = xi[:-1, :-1].multiply(theta1 * prob_off).sum()  # theta_2

    # off map transitions

    tot1 += xi[:-1, -1].toarray()[:, 0].dot((opt_dev - theta2) * prob_off)
    tot2 -= xi[:-1, -1].toarray()[:, 0].dot(prob_off)

    return -np.array([tot1, tot2])


def l_theta3(theta3, xi, source, dest, dev_within):

    N = xi.shape[0]

    E_topleft = csc_matrix((np.exp(-theta3[0] * dev_within),
                           (source, dest)), shape=(N, N))
    E_topleft = normalize(E_topleft, norm='l1', axis=1)
    E_topleft.data = np.log(E_topleft.data)

    tot = xi.multiply(E_topleft).sum()
    return -tot


def compute_objective_parts(gamma, xi, prior, sims, deviations,
                            theta, kappa, Eoo, lambda1):

    T, N = gamma.shape

    obj = 0.

    for t in range(T-1):
        dev_within, opt_dev = deviations[t]["dev"]
        source_indices = deviations[t]["source"]
        dest_indices = deviations[t]["dest"]

        obj -= l_theta12(theta[t, :2], xi[t], opt_dev)
        obj -= l_theta3(np.array([theta[t, 2]]), xi[t], source_indices, dest_indices, dev_within)

        # add from off-map transitions

        obj += xi[t][-1, :-1].sum() * (np.log(1. - Eoo) - np.log(N - 1))
        obj += xi[t][-1, -1] * np.log(Eoo)

    # add observation lhood relevant terms

    obj += ((kappa[:, None] * sims - lambda1 / 2. * kappa[:, None] ** 2) *
            gamma).sum()

    # add prior terms

    obj += (gamma[0] * np.log(prior)).sum()

    return obj


def Mstep(gamma, xi, prior, sims, deviation, theta, kappa, Eoo, lambda1):

    T = len(gamma)

    theta_new = theta.copy()
    kappa_new = kappa.copy()

    Eoo_new_num = 0.
    Eoo_new_denom = 0.

    for t in range(T-1):

        dev_within, dev_off = deviation[t]["dev"]
        source_indices = deviation[t]["source"]
        dest_indices = deviation[t]["dest"]

        bound = Bounds(np.array([0., 0.]), np.array([np.inf, np.inf]))
        bound1 = Bounds(0., np.inf)
        res = minimize(l_theta12, theta[t, :2], args=(xi[t], dev_off),
                       method='L-BFGS-B', bounds=bound)
        res1 = minimize(l_theta3, np.array([theta[t, 2]]),
                        args=(xi[t], source_indices, dest_indices, dev_within),
                        method='L-BFGS-B', bounds=bound1)
                        # jac=l_theta12_jac
                       # options={"iprint": 99})
        theta_new[t, :2] = res.x
        theta_new[t, 2] = res1.x

        Eoo_new_num += xi[t][-1, -1]
        Eoo_new_denom += xi[t][-1].sum()

    kappa_new = (sims * gamma).sum(axis=1) / lambda1
    Eoo_new = Eoo_new_num / Eoo_new_denom
    #Eoo_new = Eoo
    return theta_new, kappa_new, Eoo_new


def bayes_recursion(vpr_lhood, transition_matrix, off_map_prob, prior_belief,
                      prior_off_classif, initial=False):
    """
    update posterior belief

    NOTE: If initial=True, alpha_prev must be prior belief at time 0
    """
    # compute factor used to rescale inverse sensor (off-map) measurements

    # compute prior belief after motion update
    if not initial:
        prediction = transition_matrix.T @ prior_belief
    else:
        prediction = prior_belief
    # perform posterior update with new off-map classification
    r1 = prior_off_classif / (1. - prior_off_classif)
    r2 = (1. - off_map_prob) / off_map_prob
    r3 = (1 - prior_belief[-1]) / prior_belief[-1]
    updated_belief_off = 1. / (1. + r1 * r2 * r3)  # p(x_t = off | z_{1:t})
    # compute scale factor for new forward lhood
    scale_factor = prior_belief[:-1] @ vpr_lhood / (1. - updated_belief_off)

    # compute recursion

    lhood_off = updated_belief_off * scale_factor / prior_belief[-1]
    lhoods = np.append(vpr_lhood, lhood_off)
    posterior_belief = prediction * lhoods
    posterior_belief /= posterior_belief.sum()

    return posterior_belief


def online_localization(odom, vpr_lhood, prior_off_classif, off_map_probs, prior,
                        ref_map, Eoo, theta1, theta2, theta3, w):
    win = 5
    posterior = prior.copy()
    for t in range(len(odom)):
        # check convergence
        ind_max = np.argmax(posterior[:-2])
        wind = posterior[max(0, ind_max - win):min(len(posterior) - 2, ind_max + win)]
        score = wind.sum()
        print(t, ind_max, score)
        if score > 0.3:
            import matplotlib.pyplot as plt
            plt.bar(np.arange(len(posterior)-1), posterior[:-1])
            plt.show()
            return t, ind_max
        # compute stuff for bayes recursion
        dev = create_deviation_matrix(ref_map, odom[t], Eoo, w)
        E = create_transition_matrix(dev, len(ref_map), Eoo, theta1, theta2, theta3)
        # Bayes recursion
        if t == 0:
            posterior = bayes_recursion(vpr_lhood[t], E, off_map_probs[t],
                                        posterior, prior_off_classif, initial=True)
        else:
            posterior = bayes_recursion(vpr_lhood[t], E, off_map_probs[t],
                                        posterior, prior_off_classif, initial=True)
    # localization failure (failed to localize before EOS)
    return False
