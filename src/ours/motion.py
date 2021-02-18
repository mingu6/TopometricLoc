import numpy as np
from scipy.stats import chi2
import scipy.sparse as sparse
from scipy.sparse import csc_matrix, coo_matrix
from sklearn.preprocessing import normalize


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

        t = ((segl - mu)^T Sigma^-1 (selu - segl)) /
                    ((segu - segl)^T Sigma^-1 (segu - segl))
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


def transition_probs(qmu, qSigma, refMap, p_off_max):
    """
    Transition probabilities between within map nodes and also from map
    node to off-map state.
    """
    N = refMap.N
    wd = refMap.width
    # compute minimum MH distance between query and segments
    segl = refMap.odom_segments[..., 0, :].reshape(-1, 3)
    segu = refMap.odom_segments[..., 1, :].reshape(-1, 3)
    sqdis = min_MN_dist_seg(qmu, qSigma, segl, segu).reshape(N, wd+1)
    sqdis = np.clip(sqdis, 0., 9.)
    # compute relative likelihoods odomd for transition probabilities
    # to within map nodes
    qlhood = np.exp(-0.5 * sqdis)
    qlhood /= qlhood.sum(axis=1)[:, None]
    # apply chisq cdf to compute within map prob. Evaluating cdf is same as
    # prob sqmndist <= x. Intuitively, P(sqMN <= x) gives an indication of
    # off-map likelihood between 0, 1
    off_prob = (chi2.cdf(sqdis, 3) * p_off_max * qlhood).sum(axis=1)
    within_prob = qlhood * (1. - off_prob)[:, None]
    return within_prob, off_prob
