import numpy as np

import scipy.sparse as sparse
from sklearn.preprocessing import normalize

from motion import odom_deviation


def create_transition_matrix(qOdom, odom_segments, att_wt, theta):
    devs = odom_deviation(qOdom, odom_segments, att_wt)
    data = np.exp(-theta * devs)
    N, wd, _, _ = odom_segments.shape
    # construct sparse matrix
    mat = sparse.diags(data.T, offsets=np.arange(wd), shape=(N, N),
                       format="csr", dtype=np.float32)
    normalize(mat, norm='l1', axis=1, copy=False)
    return mat.tocsc()

def create_transition_matrix(qOdom, odom_segments, att_wt, sigma):
    devs = odom_deviation(qOdom, odom_segments, att_wt)
    data = np.exp(-devs ** 2 / (2 * sigma ** 2))
    N, wd, _, _ = odom_segments.shape
    # construct sparse matrix
    mat = sparse.diags(data.T, offsets=np.arange(wd), shape=(N, N),
                       format="csr", dtype=np.float32)
    normalize(mat, norm='l1', axis=1, copy=False)
    return mat.tocsc()

class Localization:
    def __init__(self, params, refMap):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # initialize belief
        N = refMap.N
        self.belief = np.ones(N) / N

        # measurement param
        self.delta = self.meas_params['delta']
        self.lambd = None

        self.refMap = refMap

        # reset parameters
        self.reset = params["other"]["reset_step"]
        self.t = 0

    def init(self, qOdom, qGlb, qLoc, uniform=False):
        """
        Allows for any initialization at time 0 before first motion update.
        Calibrates intensity parameter lambda using first VPR measurement
        and sets belief. Option to set a uniform belief initialization.
        """
        query_sims = self.refMap.glb_des @ qGlb
        dist = np.sqrt(2. - 2. * query_sims)
        descriptor_quantiles = np.quantile(dist, [0.025, 0.975])
        quantile_range = descriptor_quantiles[1] - descriptor_quantiles[0]
        self.lambd = np.log(self.delta) / quantile_range
        if not uniform:
            self.belief = np.exp(-self.lambd * dist)
            self.belief /= self.belief.sum()
        else:
            self.belief = np.ones(self.refMap.N) / self.refMap.N
        return None

    def _update_motion(self, qOdom):
        """
        Applies motion update to belief.
        """
        E = create_transition_matrix(qOdom, self.refMap.odom_segments,
                                     self.motion_params['att_wt'],
                                     self.motion_params['theta'])
        self.belief = E.T @ self.belief
        return None

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        # compute likelihoods
        query_sims = self.refMap.glb_des @ qGlb
        dist = np.sqrt(2. - 2. * query_sims)
        lhood = np.exp(-self.lambd * dist)
        self.belief *= lhood
        self.belief /= self.belief.sum()
        return None

    def update(self, qOdom, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        # reset if not converged by step limit
        if self.t >= self.reset:
            self.t = 0
            self.init(qOdom, qGlb, qLoc, uniform=True)
        # state update
        self._update_motion(qOdom)
        self._update_meas(qGlb, qLoc)
        # update step count since last reset
        self.t += 1
        return None

    def converged(self, score_thresh, nhood_size):
        window = self.other_params['convergence_window']
        ind_max = np.argmax(self.belief)
        nhood_inds = np.arange(max(ind_max-window, 0),
                               min(ind_max+window, len(self.belief)))
        belief_nhood = self.belief[nhood_inds]
        localized = belief_nhood.sum() > score_thresh
        ind_pred = round(nhood_inds.mean())
        return ind_pred, localized, None
