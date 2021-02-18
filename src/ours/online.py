import numpy as np

from scipy import sparse

from data.utils import preprocess_local_features
from ours.motion import transition_probs
from ours.measurement import measurement_update, geometric_verification


class OnlineLocalization:
    def __init__(self, params, refMap, nooff=False):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # initialize belief
        belief_size = refMap.N + 1 if not nooff else refMap.N
        self.belief = np.ones(belief_size)
        if not nooff:
            self.belief[:-1] *= (1. - self.other_params["prior_off"]) / refMap.N
            self.belief[-1] = self.other_params["prior_off"]

        # reference map
        self.refMap = refMap
        self.odom_segments = refMap.odom_segments.copy()

        # meas params
        self.lamb = None

    def init(self, qOdom, qGlb, qLoc, noverif=False, nooff=False):
        """
        Allows for any initialization at time 0 before first motion update
        """
        # calibrate measurement parameter
        query_sims = self.refMap.glb_des @ qGlb
        dist = np.sqrt(2. - 2. * query_sims)
        descriptor_quantiles = np.quantile(dist, [0.025, 0.975])
        quantile_range = descriptor_quantiles[1] - descriptor_quantiles[0]
        if self.meas_params['delta'] > 0.:
            self.lambd = np.log(self.meas_params['delta']) / quantile_range
        else:
            self.lambd = 0.
        # perform measurement update using calibrated param
        self.belief, lhood = measurement_update(self.belief, query_sims, qLoc,
                                                self.refMap, self.meas_params,
                                                self.lambd, noverif=noverif,
                                                nooff=nooff)
        return lhood

    def _update_motion(self, qOdom, nooff=False):
        """
        Applies motion update to belief.
        """
        qmu = qOdom[0]
        qSigma = qOdom[1]
        p_off_max = self.motion_params['p_off_max']
        within, off = transition_probs(qmu, qSigma, self.refMap, p_off_max)
        # prediction step for off-map state
        p_off_off = self.motion_params["p_off_off"]
        # prediction step for within map states
        N, wd = self.refMap.N, self.refMap.width
        within_transitions = sparse.diags(within.T,
                                          offsets=np.arange(wd+1),
                                          shape=(N, N),
                                          format="csc")
        if not nooff:
            within_transitions.resize((N+1, N+1))
            # transition matrix component for off to within/off probabilities
            from_off_prob = np.ones(N+1) * (1. - p_off_off) / N
            from_off_prob[-1] = p_off_off
            from_off_transitions = sparse.csc_matrix(
                (from_off_prob, (np.ones(N+1) * N, np.arange(N+1))), shape=(N+1, N+1))
            # transition matrix component for to off probabilities
            off += 1. - off - np.asarray(within_transitions.sum(axis=1))[:-1, 0]
            to_off_transitions = sparse.csc_matrix(
                (off, (np.arange(N), np.ones(N) * N)), shape=(N+1, N+1))
            trans_mat = within_transitions + from_off_transitions + to_off_transitions
        else:
            trans_mat = within_transitions
        self.belief = trans_mat.T @ self.belief

        return trans_mat

    def _update_meas(self, qGlb, qLoc, noverif=False, nooff=False):
        """
        Updates belief using image local and global features.
        """
        query_sims = self.refMap.glb_des @ qGlb
        self.belief, lhood = measurement_update(self.belief, query_sims, qLoc,
                                                self.refMap, self.meas_params,
                                                self.lambd, noverif=noverif,
                                                nooff=nooff)
        return lhood

    def update(self, qOdom, qGlb, qLoc, noverif=False, nooff=False):
        """
        Applies full motion and meas. update to belief.
        """
        self._update_motion(qOdom, nooff=nooff)
        self._update_meas(qGlb, qLoc, noverif=noverif, nooff=nooff)
        return None

    def converged(self, qGlb, qLoc):
        window = self.other_params['convergence_window']
        score_thres = self.other_params['convergence_score']

        # take window around posterior mode, check prob. mass underneath

        sum_belief = np.convolve(self.belief[:-1], np.ones(2 * window + 1),
                                 mode='same')
        ind_max = np.argmax(sum_belief)
        ind_max = np.argmax(self.belief[:-1])
        score = sum_belief[ind_max] / (1. - self.belief[-1])
        #score = sum_belief[ind_max]
        localized = score > score_thres \
                 and self.belief[-1] < self.other_params['off_map_lb']
        #localized = score > score_thres
        # ind_max = np.argmax(self.belief[:-1])
        # nhood_inds = np.arange(max(ind_max-window, 0),
                               # min(ind_max+window, len(self.belief)-1))
        # belief_nhood = self.belief[nhood_inds] / (1. - self.belief[-1])
        # # if belief concentrated and off-map prob is low, converged
        # converged = (belief_nhood.sum() > score_thres) and \
            # (self.belief[-1] < self.other_params['off_map_lb'])
        # ind_pred = round(nhood_inds.mean())  # proposal weighted by belief

        # # check that only one mode exists, identify next largest mode

        # belief_alt = self.belief[:-1].copy()
        # larger_window = np.arange(max(ind_pred-window*3, 0),
                                  # min(ind_pred+window*3, len(self.belief)-1))
        # belief_alt[larger_window] = 0.
        # ind_pred_next = np.argmax(belief_alt)
        # nhood_inds_next = np.arange(max(ind_pred_next-window, 0),
                                    # min(ind_pred_next+window, len(self.belief)-1))
        # belief_nhood_next = belief_alt[nhood_inds_next] / (1. - self.belief[-1])

        # # if second mode exists (with meaningful mass), set 0 score so model
        # # does not converge upon computing curves used in results

        # score = belief_nhood.sum()
        # localized = converged
        # if belief_nhood_next.sum() > 0.05:
            # converged = False
            # score = 0.

        # if filter has converged, perform geometric verification to
        # proposed place and accept location hypothesis if succeeds

        # localized = False
        # if converged:
            # meas_params = self.meas_params
            # nFeats = meas_params['num_feats']
            # qkp, qdes = preprocess_local_features(qLoc, nFeats)
            # rkp, rdes = self.refMap.load_local(ind_pred, nFeats)
            # verified = geometric_verification(rkp, rdes, qkp, qdes,
                                              # meas_params['num_inliers'],
                                              # meas_params['inlier_threshold'],
                                              # meas_params['confidence'])
            # if verified:
                # localized = True
            # else:
                # self.init(None, qGlb, qLoc)

        return ind_max, localized, score
