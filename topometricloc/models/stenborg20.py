import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

from ..geometry import SE2
from .localization_base import LocalizationBase


def create_transition_matrix(odom_mu, odom_segments, sigma):
    """
    Create state transition matrix using relative pose information
    from reference and query odom
    Args:
        odom_mu: Relative odom between prev. and curr. query image (np 3)
        odom_segments: N x (width+1) matrix of reference fw odom
                       representing state transitions from each node
    Returns:
        state transition matrix: sp sparse
    """
    N, width = odom_segments.shape
    prob = np.exp(- (odom_mu[0] - odom_segments) ** 2 / (2 * sigma ** 2))
    mat = sparse.diags(prob.T, offsets=np.arange(width), shape=(N, N), format="csr", dtype=np.float64)
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
    relpose = np.zeros((N, width+1), dtype=np.float64)
    # for each source node/connection compute relative pose
    agg_trans = SE2(np.zeros((N, 3)))
    for w in range(1, width+1):
        agg_trans = agg_trans[:-1] * SE2(odom[w:])
        relpose[:-w, w] = agg_trans.to_vec()[:, 0]
    return relpose


class Localization(LocalizationBase):
    def __init__(self, params, ref_map):
        super().__init__(params, ref_map)

        self.sigma_motion = self.motion_params['sigma']
        self.sigma_meas = self.meas_params['sigma']

        self.belief = np.ones(ref_map.N) / ref_map.N
        self.odom_segments = odom_segments(ref_map.odom, self.motion_params['width'])
        self.ref_map = ref_map

    def init(self, global_desc):
        lhood = self._update_meas(global_desc)
        return lhood

    def _update_motion(self, odom_mu, odom_sigma):
        E = create_transition_matrix(odom_mu, self.odom_segments, self.sigma_motion)
        self.belief = E.T @ self.belief
        return E

    def _update_meas(self, global_desc):
        query_sims = self.ref_map.glb_des @ global_desc
        dist_sq = 2. - 2. * query_sims
        lhood = np.exp(- (dist_sq) / (2 * self.sigma_meas ** 2))
        self.belief *= lhood
        self.belief /= self.belief.sum()
        return lhood
