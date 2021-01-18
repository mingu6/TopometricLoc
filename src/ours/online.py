import numpy as np

from scipy import sparse

from data.utils import preprocess_local_features
from ours.motion import odom_deviation, transition_probs
from ours.measurement import measurement_update, geometric_verification


class OnlineLocalization:
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

        # meas params
        self.lamb = None

    def init(self, qOdom, qGlb, qLoc):
        """
        Allows for any initialization at time 0 before first motion update
        """
        query_sims = self.refMap.glb_des @ qGlb
        dist = np.sqrt(2. - 2. * query_sims)
        descriptor_quantiles = np.quantile(dist, [0.025, 0.975])
        quantile_range = descriptor_quantiles[1] - descriptor_quantiles[0]
        if self.meas_params['delta'] > 0.:
            self.lambd = np.log(self.meas_params['delta']) / quantile_range
        else:
            self.lambd = 0.
        self.belief[:-1] = np.exp(-self.lambd * dist)
        self.belief[:-1] *= (1. - self.belief[-1]) / self.belief[:-1].sum()
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
        p_off_on = (1. - p_off_off) / N * self.belief[-1]
        self.belief[:-1] += p_off_on
        # first few nodes receive extra mass from off-map state because they
        # have more out connections than in connections. Causes huge decrease
        # in belief of these states, which affects other states over time
        self.belief[:wd] += np.flip(np.arange(1, wd+1)) * p_off_on
        self.belief[:-1] *= (1. - off_new) / self.belief[:-1].sum()
        self.belief[-1] = off_new
        return None

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        query_sims = self.refMap.glb_des @ qGlb
        self.belief = measurement_update(self.belief, query_sims, qLoc,
                                         self.refMap, self.meas_params,
                                         self.lambd)
        return None

    def update(self, qOdom, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        self._update_motion(qOdom)
        self._update_meas(qGlb, qLoc)
        return None

    def converged(self, qGlb, qLoc):
        window = self.other_params['convergence_window']
        score_thres = self.other_params['convergence_score']

        # take window around posterior mode, check prob. mass underneath

        ind_max = np.argmax(self.belief[:-1])
        nhood_inds = np.arange(max(ind_max-window, 0),
                               min(ind_max+window, len(self.belief)-1))
        belief_nhood = self.belief[nhood_inds] / (1. - self.belief[-1])
        # if belief concentrated and off-map prob is low, converged
        converged = (belief_nhood.sum() > score_thres) and \
            (self.belief[-1] < self.other_params['off_map_lb'])
        ind_pred = round(nhood_inds.mean())  # proposal weighted by belief

        # check that only one mode exists, identify next largest mode

        belief_alt = self.belief[:-1].copy()
        larger_window = np.arange(max(ind_pred-window*3, 0),
                                  min(ind_pred+window*3, len(self.belief)-1))
        belief_alt[larger_window] = 0.
        ind_pred_next = np.argmax(belief_alt)
        nhood_inds_next = np.arange(max(ind_pred_next-window, 0),
                                    min(ind_pred_next+window, len(self.belief)-1))
        belief_nhood_next = belief_alt[nhood_inds_next] / (1. - self.belief[-1])

        # if second mode exists (with meaningful mass), set 0 score so model
        # does not converge upon computing curves used in results

        score = belief_nhood.sum()
        if belief_nhood_next.sum() > 0.05:
            converged = False
            score = 0.

        # if filter has converged, perform geometric verification to
        # proposed place and accept location hypothesis if succeeds

        localized = False
        if converged:
            meas_params = self.meas_params
            nFeats = meas_params['num_feats']
            qkp, qdes = preprocess_local_features(qLoc, nFeats)
            rkp, rdes = self.refMap.load_local(ind_pred, nFeats)
            verified = geometric_verification(rkp, rdes, qkp, qdes,
                                              meas_params['num_inliers'],
                                              meas_params['inlier_threshold'],
                                              meas_params['confidence'])
            if verified:
                localized = True
            else:
                self.init(None, qGlb, qLoc)

        return ind_max, localized, score
