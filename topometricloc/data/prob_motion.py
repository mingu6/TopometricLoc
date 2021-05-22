from collections import namedtuple
import numpy as np

from ..geometry import SE2, wrapAngle


def to_banana(xytheta):
    """
    Converts vectors of form (x, y, theta) in final dimension with shape (..., 3)
    to banana distribution parameterization consisting of (rot1, transl, rot2)
    """
    rot1 = np.arctan2(xytheta[..., 1], xytheta[..., 0])
    trans = np.linalg.norm(xytheta[..., :2], axis=-1)
    rot2 = wrapAngle(xytheta[..., 2] - rot1)
    rot2 = rot2.reshape(rot1.shape)
    return np.concatenate((rot1[..., None], trans[..., None], rot2[..., None]),
                          axis=-1)


def apply_odometry(u_t, x_t_1):
    """
    Apply odometry u_t to previous pose x_{t-1}
    """
    return (SE2(x_t_1) * SE2(u_t)).to_vec()


def banana_covar(params, banana, inverse=False):
    """
    Turn odometry in "banana" form, i.e. (rot1, trans, rot2) into covariance
    matrix in control or "banana" space.
    Args:
        params: alpha1, ..., alpha4, proportional covariance params
        banana: 3-vector (..., rot1, trans, rot2) converted from x, y, theta
    Return:
        covar: 3-vector of diagonal covariance matrix
    """
    assert banana.shape[-1] == 3
    assert params.shape == (4,)

    rot1, trans, rot2 = banana.T
    alpha1, alpha2, alpha3, alpha4 = params

    # compute covariance diagonal

    covrot1 = np.clip(alpha1 * rot1 ** 2 + alpha2 * trans ** 2, 1e-3, np.inf)
    covtrans = np.clip(alpha3 * trans ** 2 + alpha4 * (rot1 ** 2 + rot2 ** 2),
                       1e-3, np.inf)
    covrot2 = np.clip(alpha1 * rot2 ** 2 + alpha2 * trans ** 2, 1e-3, np.inf)

    cov_vec = np.squeeze(np.vstack((covrot1, covtrans, covrot2)).T)
    if inverse:
        cov_vec = cov_vec ** -1
    return cov_vec


def odometry_linearize_u(u_t, mu_t_1):
    """
    Linearize motion model in apply_odometry function w.r.t. odom u_t
    (in banana form) around point mu_{t-1}, yielding a Jacobian matrix.
    Args:
        u_t: odometry at time t (3-vector, with x, y, theta)
        mu_t_1: linearlization point (3-vector, with x, y, theta)
    Returns:
        V_t: 3x3 Jacobian, columns corresp. to state (x, y, theta),
             rows to banana coords (rot1, trans, rot2).
    """
    assert u_t.shape == (3,)
    assert mu_t_1.shape == (3,)

    rot1, trans, rot2 = to_banana(u_t)
    x, y, theta = mu_t_1

    # intermediate calcs

    s = np.sin(theta + rot1)
    c = np.cos(theta + rot1)

    V_t = np.array([[- trans * s, c,  0.],
                    [  trans * c, s,  0.],
                    [1.         , 0., 1.]])
    return V_t


def odometry_linearize_x(u_t, mu_t_1):
    """
    Linearize motion model in apply_odometry function w.r.t. state x_t
    around point mu_{t-1}, yielding a Jacobian matrix.
    Args:
        u_t: odometry at time t (3-vector, with x, y, theta)
        mu_t_1: linearlization point (3-vector, with x, y, theta)
    Returns:
        G_t: 3x3 Jacobian, columns corresp. to updated state (x, y, theta),
             rows to prev. state (x, y, theta).
    """
    assert u_t.shape == (3,)
    assert mu_t_1.shape == (3,)

    rot1, trans, rot2 = to_banana(u_t)
    x, y, theta = mu_t_1

    G_t = np.array([[1., 0., -trans * np.sin(theta + rot1)],
                    [0., 1.,  trans * np.cos(theta + rot1)],
                    [0., 0.,  0.                          ]])
    return G_t


def motion_update(params, u_t, mu_t_1, Sigma_t_1):
    """
    Apply EKF filtering update step.
    Args:
        params: Covariance params alpha1, ..., alpha4 for odom noise
        u_t: Odometry containing relative pose in local frame (x, y, theta)
        mu_t_1: Mean of posterior at prev. time step
        Sigma_t_1: Covariance matrix of posterior at prev. time step
    """
    mu_t = apply_odometry(u_t, mu_t_1)
    V_t = odometry_linearize_u(u_t, mu_t_1)
    G_t = odometry_linearize_x(u_t, mu_t_1)
    M_t = np.diag(banana_covar(params, to_banana(u_t)))
    Sigma_t = G_t @ Sigma_t_1 @ G_t.T + V_t @ M_t @ V_t.T
    return mu_t, Sigma_t
