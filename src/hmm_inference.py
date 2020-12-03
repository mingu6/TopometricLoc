import numpy as np

from motion_model import create_transition_matrix


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


def online_localization(deviations, vpr_lhood, prior_off_classif, off_map_probs, prior,
                        Eoo, theta1, theta2, theta3, xyzrpy):
    win = 5
    posterior = prior.copy()
    for t in range(len(deviations)):
        # check convergence
        ind_max = np.argmax(posterior[:-2])
        wind = np.arange(max(0, ind_max - win), min(len(posterior) - 2, ind_max + win))
        score = posterior[wind].sum()
        #print(t, ind_max, score, posterior[-1])
        if score > 0.1:
            # import matplotlib.pyplot as plt
            # plt.bar(np.arange(len(posterior)-1), posterior[:-1])
            # plt.show()
            ind_final = int(np.average(wind, weights=posterior[wind]))
            return t, ind_final, posterior
        # compute stuff for bayes recursion
        E = create_transition_matrix(deviations[t], len(prior), Eoo, theta1, theta2, theta3)
        # Bayes recursion
        if t == 0:
            posterior = bayes_recursion(vpr_lhood[t], E, off_map_probs[0],
                                        posterior, prior_off_classif, initial=True)
        else:
            posterior = bayes_recursion(vpr_lhood[t], E, off_map_probs[0],
                                        posterior, prior_off_classif, initial=False)
    # TODO: properly handle localization failure (failed to localize before EOS)
    return False, False, False
