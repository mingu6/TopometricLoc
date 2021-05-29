import numpy as np
from scipy.stats import chi2
from scipy.sparse import csc_matrix, diags
from sklearn.preprocessing import normalize

from .xu20topo import calibrate
from .localization_base import LocalizationBase


def min_MN_dist_seg(mu, Sigma, segl, segu):
    """
    Minimum Mahalanobis distance of points lying on a line segment encoded
    by the lower and upper segment points (segl, segu) for Gaussian of
    mean mu and covariance Sigma.
    Args:
        mu: len d mean vector
        Sigma: dxd covariance matrix
        segl: Nxd segment start points
        segu: Nxd segment end points
    Returns:
        sqmndist: len N vector containing minimum sq MN dist to segments

    NOTE:
        t \in [0, 1] is the location within the segment where MN is minimized
        given by:

        t = ((segl - mu)^T Sigma^-1 (selu - segl)) / ((segu - segl)^T Sigma^-1 (segu - segl))
            and subsequently restricted between 0 and 1
    """
    Sigmainv = np.linalg.inv(Sigma)
    segdiff = (segu - segl)
    segdifftransf = segdiff @ Sigmainv

    # interpolation point for each segment
    denom = (segdifftransf * segdiff).sum(axis=1)
    t = ((mu[None, :] - segl) * segdifftransf).sum(axis=1)
    t[denom != 0.] /= denom[denom != 0.]
    t = np.clip(t, 0., 1.)[:, None]

    # compute optimal point and distance
    opt_pt = segu * t + (1. - t) * segl
    diff = mu - opt_pt
    sqmndist = ((diff @ Sigmainv) * diff).sum(axis=1)
    return sqmndist


def transition_probs(odom_mu, odom_sigma, ref_map, p_off_max):
    """
    Transition probabilities between within map nodes and also from map
    node to off-map state.
    """
    N, wd = ref_map.N, ref_map.width
    # compute minimum MH distance between query and segments
    segl = ref_map.odom_segments[..., 0, :].reshape(-1, 3)
    segu = ref_map.odom_segments[..., 1, :].reshape(-1, 3)
    #import pdb; pdb.set_trace()
    #sqdis = min_MN_dist_seg(odom_mu, odom_sigma, segl, segu).reshape(N, wd+1)
    sqdis = min_MN_dist_seg(odom_mu, odom_sigma, segl, segu).reshape(N, -1)
    sqdis = np.clip(sqdis, 0., 9.)
    # compute relative likelihoods odomd for transition probabilities
    # to within map nodes
    qlhood = np.exp(-0.5 * sqdis)
    qlhood /= qlhood.sum(axis=1)[:, None]
    # apply chisq cdf to compute within map prob. Evaluating cdf is same as
    # prob sqmndist <= x. Intuitively, P(sqMN <= x) gives an indication of
    # off-map likelihood between 0, 1
    off_prob = np.clip(chi2.cdf(sqdis.min(axis=1), 3) * p_off_max, 0.02, 1.).astype(np.float32)
    within_prob = qlhood * (1. - off_prob)[:, None]
    return within_prob, off_prob


def within_lhood(global_query, global_reference, lambd):
    query_sims = global_reference @ global_query
    dist = np.sqrt(2. - 2. * query_sims)
    lhood = np.exp(-lambd * dist)
    return lhood


class Localization(LocalizationBase):
    """
    Our full topometric localization model as described in the paper.
    """
    def __init__(self, params, ref_map):
        super().__init__(params, ref_map)
        self.off_state = self.other_params["off_state"]
        N = self.ref_map.N + 1 if self.off_state else self.ref_map.N
        self.belief = np.ones(N, dtype=np.float32)
        if self.off_state:
            self.belief[:-1] *= (1. - self.other_params["prior_off"]) / ref_map.N
            self.belief[-1] = self.other_params["prior_off"]

        self.odom_segments = ref_map.odom_segments.copy()

        self.lamb = None
        self.off_k = 20

    def init(self, global_desc):
        """
        Performs calibration at initial timestep
        """
        delta = self.meas_params['delta']
        self.lambd, lhood = calibrate(global_desc, self.ref_map, delta)
        if self.off_state:
            off_lhood = np.partition(lhood, self.off_k)[self.off_k]
            #off_lhood = 1 / (1. - self.belief[-1]) * lhood @ self.belief[:-1]
            lhood = np.hstack((lhood, off_lhood))
        self.belief = self.belief * lhood
        self.belief /= self.belief.sum()
        return lhood

    def _update_motion(self, odom_mu, odom_sigma):
        N, wd = self.ref_map.N, self.ref_map.width
        p_off_max = self.motion_params['p_off_max']
        # compute transition probabilities from within-map states
        within, off = transition_probs(odom_mu, odom_sigma, self.ref_map, p_off_max)
        # prediction step for off-map state
        p_off_off = self.motion_params["p_off_off"]
        # transition probabilities for within map states
        trans_mat = diags(within.T, offsets=np.arange(wd+1), shape=(N, N), format="csc")
        if self.off_state:
            # incorporate transitions to/from off-map state in matrix
            trans_mat.resize((N+1, N+1))
            # transition matrix component for from off-map state
            from_off_prob = np.ones(N+1, dtype=np.float32) * (1. - p_off_off) / N
            from_off_prob[-1] = p_off_off
            from_off_transitions = csc_matrix((from_off_prob, (np.ones(N+1, dtype=np.float32) * N, np.arange(N+1))), shape=(N+1, N+1))
            # transition matrix component for to off probabilities
            off += 1. - off - np.asarray(trans_mat.sum(axis=1))[:-1, 0]
            to_off_transitions = csc_matrix((off, (np.arange(N), np.ones(N) * N)), shape=(N+1, N+1))
            trans_mat = trans_mat + from_off_transitions + to_off_transitions
        self.belief = trans_mat.T @ self.belief
        return trans_mat

    def _update_meas(self, global_desc):
        lhood = within_lhood(global_desc, self.ref_map.glb_des, self.lambd)
        if self.off_state:
            #off_lhood = 1 / (1. - self.belief[-1]) * lhood @ self.belief[:-1]
            off_lhood = np.partition(lhood, self.off_k)[self.off_k]
            lhood = np.hstack((lhood, off_lhood))
        self.belief = self.belief * lhood
        self.belief /= self.belief.sum()
        return lhood
