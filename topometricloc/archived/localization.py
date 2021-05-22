import numpy as np

from scipy import sparse

from motion import odom_deviation, transition_probs
from measurement import measurement_update


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
        self.odom_segments = refMap.odom_segments.copy()
        #self.odom_segments[..., -1] *= self.motion_params["att_wt"]

    def init(self, qOdom, qGlb, qLoc):
        """
        Allows for any initialization at time 0 before first motion update
        """
        self._update_meas(qGlb, qLoc)
        return None

    def _update_motion(self, qOdom):
        """
        Applies motion update to belief.
        """
        att_wt = self.motion_params["att_wt"]
        # compute deviations and within -> within/off probabilities
        dev = odom_deviation(qOdom, self.odom_segments, att_wt)
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
        #self.belief[-1] = 0.
        #self.belief[:-1] /= self.belief[:-1].sum()
        return None

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        query_sims = self.refMap.glb_des @ qGlb
        self.belief = measurement_update(self.belief, query_sims, qLoc,
                                         self.refMap, self.meas_params)
        #self.belief[:-1] /= self.belief[:-1].sum()
        #self.belief[-1] = 0.
        return None

    def update(self, qOdom, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        self._update_motion(qOdom)
        self._update_meas(qGlb, qLoc)
        return None

    def converged(self, score_thresh, nhood_size):
        nhood = np.ones(2 * nhood_size + 1)
        scores = np.convolve(self.belief[:-1], nhood, mode='same')
        ind_max = np.argmax(scores)  # nhood with most prob. mass
        localized = scores[ind_max] / (1. - self.belief[-1]) > score_thresh
        localized = localized and (self.belief[-1] < 0.4)

        # window = self.other_params['convergence_window']
        # ind_max = np.argmax(self.belief[:-1])
        # nhood_inds = np.arange(max(ind_max-window, 0),
                               # min(ind_max+window, len(self.belief)-1))
        # belief_nhood = self.belief[nhood_inds] / (1. - self.belief[-1])
        # localized = (belief_nhood.sum() > score_thresh) and (self.belief[-1] < 0.4)
        # ind_max = round(nhood_inds.mean())

        # scores = np.zeros(self.refMap.N)
        # scores[ind_max] = belief_nhood.sum()
        # return ind_max, localized, scores
        return ind_max, localized, scores
