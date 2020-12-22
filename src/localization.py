import numpy as np

from scipy import sparse

from motion import odom_deviation, transition_probs


class Localization:
    def __init__(self, params, refMap):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # initialize belief
        self.belief = np.ones(refMap.N + 1)
        self.belief[:-1] *= (1. - self.other_params["prior_off"]) / refMap.N
        self.belief[-1] = self.other_params["prior_off"]

        # reference map
        self.refMap = refMap
        self.refMap.odom_segments[..., -1] *= self.motion_params["att_wt"]

    def _update_motion(self, qOdom):
        """
        Given query odometry, create transition matrix and
        update belief
        """
        att_wt = self.motion_params["att_wt"]
        # compute deviations and within -> within/off probabilities
        dev = odom_deviation(qOdom, self.refMap, att_wt)
        within, off = transition_probs(dev,
                                       self.motion_params["p_off_min"],
                                       self.motion_params["p_off_max"],
                                       self.motion_params["d_min"],
                                       self.motion_params["d_max"],
                                       self.motion_params["theta"])
        # prediction step for off-map state
        p_off_off = self.motion_params["p_off_off"]
        off_new = off.dot(self.belief[:-1]) + self.belief[-1] * p_off_off
        # prediction step for within map states
        N, wd = self.refMap.N, self.refMap.width
        within_transitions = sparse.diags(within.T,
                                          offsets=np.arange(wd+1),
                                          shape=(N, N),
                                          format="csc",
                                          dtype=np.float32)
        self.belief[:-1] = within_transitions.T @ self.belief[:-1]
        # off to within transition
        self.belief[:-1] += (1. - p_off_off) / N * self.belief[-1]

        self.belief[-1] = off_new
        return None

    def _update_meas(self, qGlb, qLoc):
        self.belief 

    def update(self, qOdom, qGlb, qLoc):
        asdas

    def converged(self):
        asdas


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
                        Eoo, theta, p_min, p_max, d_min, d_max, width, xyzrpy):
    win = 5
    posterior = prior.copy()
    params = {"Eoo": Eoo, "theta": theta, "N": len(prior), "p_min": p_min,
              "p_max": p_max, "d_min": d_min, "d_max": d_max, "width": width}
    for t in range(len(deviations)):
        # check convergence
        ind_max = np.argmax(posterior[:-1])
        wind = np.arange(max(0, ind_max - win), min(len(posterior) - 1, ind_max + win))
        score = posterior[wind].sum()
        #print(t, ind_max, score, posterior[-1])
        if score > 0.1:
            # import matplotlib.pyplot as plt
            # plt.bar(np.arange(len(posterior)-1), posterior[:-1])
            # plt.show()
            ind_final = int(np.average(wind, weights=posterior[wind]))
            return t, ind_final, posterior
        # compute stuff for bayes recursion
        E = create_transition_matrix(deviations[t], params)
        # Bayes recursion
        if t == 0:
            posterior = bayes_recursion(vpr_lhood[t], E, off_map_probs[0],
                                        posterior, prior_off_classif, initial=True)
        else:
            posterior = bayes_recursion(vpr_lhood[t], E, off_map_probs[0],
                                        posterior, prior_off_classif, initial=False)
    # TODO: properly handle localization failure (failed to localize before EOS)
    return False, False, False
