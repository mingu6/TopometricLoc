import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

from .localization_base import LocalizationBase


def calibrate(global_desc, ref_map, delta):
    query_sims = ref_map.glb_des @ global_desc
    dist = np.sqrt(2. - 2. * query_sims)
    descriptor_quantiles = np.quantile(dist, [0.025, 0.975])
    quantile_range = descriptor_quantiles[1] - descriptor_quantiles[0]
    if delta > 0.:
        lambd = np.log(delta) / quantile_range
    else:
        lambd = 0.
    lhood = np.exp(-lambd * dist)
    return lambd, lhood


def create_transition_matrix(N, lower, upper):
    """
    Create state transition matrix with uniform transition probabilities.
    Args:
        N: Number of states/places
        lower: index of earliest relative transition. e.g. -2 means given a
               state, a transition of -2 nodes is possible.
        upper: index of furthest relative transition
    """
    assert upper > lower
    data = np.ones((upper - lower, N)) / (upper - lower)
    mat = sparse.diags(data, offsets=np.arange(lower, upper), shape=(N, N), format="csr", dtype=np.float64)
    mat = normalize(mat, norm='l1', axis=1, copy=False)
    return mat.tocsc()


class Localization(LocalizationBase):
    def __init__(self, params, ref_map):
        super().__init__(params, ref_map)

        self.belief = np.ones(ref_map.N) / ref_map.N
        self.E = create_transition_matrix(ref_map.N, self.motion_params['lower'], self.motion_params['upper'])

        self.delta = self.meas_params['delta']
        self.lambd = None

    def init(self, global_desc):
        self.lambd, lhood = calibrate(global_desc, self.ref_map, self.delta)
        self.belief = lhood
        self.belief /= self.belief.sum()
        return lhood

    def _update_motion(self, odom_mu, odom_sigma):
        self.belief = self.E.T @ self.belief
        return self.E

    def _update_meas(self, global_desc):
        query_sims = self.ref_map.glb_des @ global_desc
        dist = np.sqrt(2. - 2. * query_sims)
        lhood = np.exp(-self.lambd * dist)
        self.belief *= lhood
        self.belief /= self.belief.sum()
        return lhood
