import numpy as np

import scipy.sparse as sparse
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

from geometry import SE2


def create_transition_matrix(qmu, odom_segments, sigma):
    """
    Create state transition matrix using relative pose information
    from reference and query odom
    Args:
        qmu: Relative odom between prev. and curr. query image (np 3)
        odom_segments: N x (width+1) matrix of reference fw odom
                       representing state transitions from each node
    Returns:
        state transition matrix: sp sparse
    """
    N, width = odom_segments.shape
    prob = np.exp(- (qmu[0] - odom_segments) ** 2 / (2 * sigma ** 2))
    mat = sparse.diags(prob.T, offsets=np.arange(width), shape=(N, N),
                       format="csr", dtype=np.float32)
    normalize(mat, norm='l1', axis=1, copy=False)
    return mat.tocsc()


def odom_segments(odom, width):
    """
    For each state transition in map, with state transitions per node given
    by width, return matrix of relative poses from each node for forward
    motion only, to connecting nodes representing state-transition
    information.

    Output should be an N x width matrix, where N is the number
    of nodes, width is the number of transitions from each node.

    Args:
        odom: Relative odometry between reference images, (N-1 x 3) np array
        width: Number of outgoing connections (state transitions) from each
        node
    Output:
        segments: N x width np array with relative pose segments as
        described above.
    """
    N = len(odom)  # N nodes has N-1 relative poses between
    # entry i, d of relpose contains relative pose from node i, i+d
    # for d <= width estimated from VO
    # note: last node has no outward connections
    relpose = np.zeros((N, width+1))
    # for each source node/connection compute relative pose
    agg_trans = SE2(np.zeros((N, 3)))
    for w in range(1, width+1):
        agg_trans = agg_trans[:-1] * SE2(odom[w:])
        relpose[:-w, w] = agg_trans.to_vec()[:, 0]
    return relpose


class Localization:
    def __init__(self, params, refMap):
        # model parameters
        self.motion_params = params["motion"]
        self.meas_params = params["measurement"]
        self.other_params = params["other"]

        # initialize belief
        N = refMap.N
        self.belief = np.ones(N) / N

        # model params
        self.sigma_motion = self.motion_params['sigma']
        self.sigma_meas = self.meas_params['sigma']

        # motion transition matrix data
        self.odom_segments = odom_segments(refMap.odom,
                                           self.motion_params['width'])

        # reference map
        self.refMap = refMap

        # reset parameters
        self.reset = params["other"]["reset_step"]
        self.t = 0

    def init(self, qmu, qSigma, qGlb, qLoc, uniform=False):
        """
        Allows for any initialization at time 0 before first motion update.
        Calibrates intensity parameter lambda using first VPR measurement
        and sets belief. Option to set a uniform belief initialization.
        """
        query_sims = self.refMap.glb_des @ qGlb
        dist_sq = 2. - 2. * query_sims
        lhood = np.exp(- dist_sq / (2 * self.sigma_meas ** 2))
        if not uniform:
            self.belief = lhood
            self.belief /= self.belief.sum()
        else:
            self.belief = np.ones(self.refMap.N) / self.refMap.N
        return lhood

    def _update_motion(self, qmu, qSigma):
        """
        Applies motion update to belief.
        """
        E = create_transition_matrix(qmu, self.odom_segments,
                                     self.sigma_motion)
        self.belief = E.T @ self.belief
        return E

    def _update_meas(self, qGlb, qLoc):
        """
        Updates belief using image local and global features.
        """
        # compute likelihoods (Gaussian lhood)
        query_sims = self.refMap.glb_des @ qGlb
        dist_sq = 2. - 2. * query_sims
        # lhood = np.exp(- (dist_sq - np.median(dist_sq)) /
                       # (2 * self.sigma_meas ** 2))
        lhood = np.exp(- (dist_sq) /
                       (2 * self.sigma_meas ** 2))
        self.belief *= lhood
        self.belief /= self.belief.sum()
        return lhood

    def update(self, qmu, qSigma, qGlb, qLoc):
        """
        Applies full motion and meas. update to belief.
        """
        # reset if not converged by step limit
        if self.t >= self.reset:
            self.t = 0
            self.init(qmu, qSigma, qGlb, qLoc, uniform=True)
        # state update
        self._update_motion(qmu, qSigma)
        self._update_meas(qGlb, qLoc)
        # update step count since last reset
        self.t += 1
        return None

    def converged(self, qGlb, qLoc):
        window = self.other_params['convergence_window']

        # take window around posterior mode, check prob. mass underneath

        sum_belief = np.convolve(self.belief, np.ones(2 * window + 1),
                                 mode='same')
        pred_ind = np.argmax(self.belief)
        score = sum_belief[pred_ind]
        check = True

        return pred_ind, check, score
