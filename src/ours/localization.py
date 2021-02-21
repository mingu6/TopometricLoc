import numpy as np

from scipy import sparse

from data.utils import preprocess_local_features
from ours.motion import transition_probs
from ours.measurement import *


def calibrate(qGlb0, refMap, delta):
    query_sims = refMap.glb_des @ qGlb0
    dist = np.sqrt(2. - 2. * query_sims)
    descriptor_quantiles = np.quantile(dist, [0.025, 0.975])
    quantile_range = descriptor_quantiles[1] - descriptor_quantiles[0]
    if delta > 0.:
        lambd = np.log(delta) / quantile_range
    else:
        lambd = 0.
    lhood = np.exp(-lambd * dist)
    return lambd, lhood


class LocalizationFull:
    """
    Our full topometric localization model, utilizing local features
    and off-map measurement and motion update.
    """
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

    def init(self, qmu, qSigma, qGlb, qLoc):
        """
        Allows for any initialization at time 0 before first motion update
        """
        # calibrate lambda, meas. lhood parameter
        delta = self.meas_params['delta']
        self.lambd, wlhood = calibrate(qGlb, self.refMap, delta)
        # update off-map likelihood
        off_lhood = off_map_lhood(qLoc, self.refMap, wlhood,
                                  self.belief, self.meas_params)
        full_lhood = np.hstack((wlhood, off_lhood))
        # update belief
        self.belief = meas_update(self.belief, full_lhood)
        return full_lhood

    def _update_motion(self, qmu, qSigma):
        """
        Applies motion update to belief.
        """
        N, wd = self.refMap.N, self.refMap.width
        p_off_max = self.motion_params['p_off_max']
        # compute transition probabilities from within-map states
        within, off = transition_probs(qmu, qSigma, self.refMap, p_off_max)
        # prediction step for off-map state
        p_off_off = self.motion_params["p_off_off"]
        # prediction step for within map states
        within_transitions = sparse.diags(within.T, offsets=np.arange(wd+1),
                                          shape=(N, N), format="csc")
        # incorporate transitions to/from off-map state in matrix
        within_transitions.resize((N+1, N+1))
        # transition matrix component for from off-map state
        from_off_prob = np.ones(N+1) * (1. - p_off_off) / N
        from_off_prob[-1] = p_off_off
        from_off_transitions = sparse.csc_matrix(
            (from_off_prob, (np.ones(N+1) * N, np.arange(N+1))), shape=(N+1, N+1))
        # transition matrix component for to off probabilities
        off += 1. - off - np.asarray(within_transitions.sum(axis=1))[:-1, 0]
        to_off_transitions = sparse.csc_matrix(
            (off, (np.arange(N), np.ones(N) * N)), shape=(N+1, N+1))
        trans_mat = within_transitions + from_off_transitions + to_off_transitions
        self.belief = trans_mat.T @ self.belief

        return trans_mat

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        # within-map lhood
        wlhood = within_lhood(qGlb, self.refMap.glb_des, self.lambd)
        # update off-map likelihood
        off_lhood = off_map_lhood(qLoc, self.refMap, wlhood,
                                  self.belief, self.meas_params)
        full_lhood = np.hstack((wlhood, off_lhood))
        # update belief
        self.belief = meas_update(self.belief, full_lhood)
        return full_lhood

    def update(self, qmu, qSigma, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        self._update_motion(qmu, qSigma)
        self._update_meas(qGlb, qLoc)
        return None

    def converged(self, qGlb, qLoc):
        """
        Convergence detection for belief.
        Args:
            qGlb: Global features (query)
            qLoc: Local features (query)
        Returns:
            ind_prop: Proposed robot place within reference map
            check: Flag to allow convergence. If check = False, do not
                   allow robot to localize even if score passes threshold
            score: Confidence score representing convergence of belief
        """
        window = self.other_params['convergence_window']
        num_feats = self.meas_params['num_feats']
        num_inliers = self.meas_params['num_inliers']
        inlier_threshold = self.meas_params['inlier_threshold']
        confidence = self.meas_params['confidence']

        # take window around posterior mode, check prob. mass underneath

        sum_belief = np.convolve(self.belief[:-1], np.ones(2 * window + 1),
                                 mode='same')
        ind_prop = np.argmax(self.belief[:-1])
        score = sum_belief[ind_prop] / (1. - self.belief[-1])
        check = self.belief[-1] < self.other_params['off_map_lb']
        if check:
            # spatial verification on most likely state
            qkp, qdes = preprocess_local_features(qLoc, num_feats)
            kp, des = self.refMap.load_local(ind_prop, num_feats)
            verif_success = geometric_verification(kp, des, qkp, qdes,
                                                   num_inliers, inlier_threshold,
                                                   confidence)
            check = check and verif_success

        return ind_prop, check, score


class LocalizationNoVerif:
    """
    Our topometric localization model, but does not utilize local features
    for the full off-map state measurement update. The off-map state
    probability does not change during the measurement update.
    """
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

    def init(self, qmu, qSigma, qGlb, qLoc):
        """
        Allows for any initialization at time 0 before first motion update
        """
        # calibrate lambda, meas. lhood parameter
        delta = self.meas_params['delta']
        self.lambd, wlhood = calibrate(qGlb, self.refMap, delta)
        # update off-map likelihood
        off_lhood = 1 / (1. - self.belief[-1]) * wlhood @ \
            self.belief[:-1]
        full_lhood = np.hstack((wlhood, off_lhood))
        # update belief
        self.belief = meas_update(self.belief, full_lhood)
        return full_lhood

    def _update_motion(self, qmu, qSigma):
        """
        Applies motion update to belief.
        """
        N, wd = self.refMap.N, self.refMap.width
        p_off_max = self.motion_params['p_off_max']
        # compute transition probabilities from within-map states
        within, off = transition_probs(qmu, qSigma, self.refMap, p_off_max)
        # prediction step for off-map state
        p_off_off = self.motion_params["p_off_off"]
        # prediction step for within map states
        within_transitions = sparse.diags(within.T, offsets=np.arange(wd+1),
                                          shape=(N, N), format="csc")
        # incorporate transitions to/from off-map state in matrix
        within_transitions.resize((N+1, N+1))
        # transition matrix component for from off-map state
        from_off_prob = np.ones(N+1) * (1. - p_off_off) / N
        from_off_prob[-1] = p_off_off
        from_off_transitions = sparse.csc_matrix(
            (from_off_prob, (np.ones(N+1) * N, np.arange(N+1))), shape=(N+1, N+1))
        # transition matrix component for to off probabilities
        off += 1. - off - np.asarray(within_transitions.sum(axis=1))[:-1, 0]
        to_off_transitions = sparse.csc_matrix(
            (off, (np.arange(N), np.ones(N) * N)), shape=(N+1, N+1))
        trans_mat = within_transitions + from_off_transitions + to_off_transitions
        self.belief = trans_mat.T @ self.belief

        return trans_mat

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        # within-map lhood
        wlhood = within_lhood(qGlb, self.refMap.glb_des, self.lambd)
        # update off-map likelihood
        off_lhood = 1 / (1. - self.belief[-1]) * wlhood @ \
            self.belief[:-1]
        full_lhood = np.hstack((wlhood, off_lhood))
        # update belief
        self.belief = meas_update(self.belief, full_lhood)
        return full_lhood

    def update(self, qmu, qSigma, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        self._update_motion(qmu, qSigma)
        self._update_meas(qGlb, qLoc)
        return None

    def converged(self, qGlb, qLoc):
        """
        Convergence detection for belief.
        Args:
            qGlb: Global features (query)
            qLoc: Local features (query)
        Returns:
            ind_prop: Proposed robot place within reference map
            check: Flag to allow convergence. If check = False, do not
                   allow robot to localize even if score passes threshold
            score: Confidence score representing convergence of belief
        """
        window = self.other_params['convergence_window']
        score_thres = self.other_params['convergence_score']
        num_feats = self.meas_params['num_feats']
        num_inliers = self.meas_params['num_inliers']
        inlier_threshold = self.meas_params['inlier_threshold']
        confidence = self.meas_params['confidence']

        # take window around posterior mode, check prob. mass underneath

        sum_belief = np.convolve(self.belief[:-1], np.ones(2 * window + 1),
                                 mode='same')
        ind_prop = np.argmax(self.belief[:-1])
        score = sum_belief[ind_prop] / (1. - self.belief[-1])
        check = self.belief[-1] < self.other_params['off_map_lb']

        return ind_prop, check, score


class LocalizationNoOff:
    """
    Our topometric localization model with no off-map state at all. Transition
    probabilities between nodes uses 3-dof odometry, but no explicit off-map
    state used for localization.
    """
    def __init__(self, params, refMap):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # initialize belief
        self.belief = np.ones(refMap.N) / refMap.N

        # reference map
        self.refMap = refMap
        self.odom_segments = refMap.odom_segments.copy()

        # meas params
        self.lamb = None

    def init(self, qmu, qSigma, qGlb, qLoc):
        """
        Allows for any initialization at time 0 before first motion update
        """
        # calibrate lambda, meas. lhood parameter
        delta = self.meas_params['delta']
        self.lambd, wlhood = calibrate(qGlb, self.refMap, delta)
        # update belief
        self.belief = meas_update(self.belief, wlhood)
        return wlhood

    def _update_motion(self, qmu, qSigma):
        """
        Applies motion update to belief.
        """
        N, wd = self.refMap.N, self.refMap.width
        # compute transition probabilities
        within, _ = transition_probs(qmu, qSigma, self.refMap, 0.)
        # prediction step
        trans_mat = sparse.diags(within.T, offsets=np.arange(wd+1),
                                 shape=(N, N), format="csc")
        self.belief = trans_mat.T @ self.belief

        return trans_mat

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        # within-map lhood
        wlhood = within_lhood(qGlb, self.refMap.glb_des, self.lambd)
        # update belief
        self.belief = meas_update(self.belief, wlhood)
        return wlhood

    def update(self, qmu, qSigma, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        self._update_motion(qmu, qSigma)
        self._update_meas(qGlb, qLoc)
        return None

    def converged(self, qGlb, qLoc):
        """
        Convergence detection for belief.
        Args:
            qGlb: Global features (query)
            qLoc: Local features (query)
        Returns:
            ind_prop: Proposed robot place within reference map
            check: Flag to allow convergence. If check = False, do not
                   allow robot to localize even if score passes threshold
            score: Confidence score representing convergence of belief
        """
        window = self.other_params['convergence_window']
        score_thres = self.other_params['convergence_score']
        num_feats = self.meas_params['num_feats']
        num_inliers = self.meas_params['num_inliers']
        inlier_threshold = self.meas_params['inlier_threshold']
        confidence = self.meas_params['confidence']

        # take window around posterior mode, check prob. mass underneath

        sum_belief = np.convolve(self.belief, np.ones(2 * window + 1),
                                 mode='same')
        ind_prop = np.argmax(self.belief)
        score = sum_belief[ind_prop]
        check = True

        return ind_prop, check, score
