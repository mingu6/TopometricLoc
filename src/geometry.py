import numpy as np
from scipy.spatial.transform import Rotation


class SE2:
    def __init__(self, xytheta):
        self._single = False

        if xytheta.ndim not in [1, 2] or xytheta.shape[-1] != 3:
            raise ValueError("Expected input containing x, y, theta values "
                             "to have shape (3,) or (N x 3), "
                             "got {}.".format(xytheta.shape))

        # If a single (x, y, theta) is given, convert it to a 2D
        # 1 x 3 matrix but set self._single to True so that we can
        # return appropriate objects in the `to_...` methods
        if xytheta.shape == (3,) or len(xytheta) == 1:
            self._single = True
        self._xy = np.atleast_2d(xytheta[..., :2])

        # ensure angles are between (-pi, pi]
        theta = np.atleast_1d(wrapAngle(xytheta[..., 2]))
        self._theta = theta

        self.len = 1 if self._single else len(xytheta)

    @classmethod
    def from_vec(cls, xytheta):
        return cls(xytheta)

    @classmethod
    def from_list(cls, SE2list):
        xytheta = np.vstack([np.hstack((s._xy, s._theta[:, None]))
                             for s in SE2list])
        return cls(xytheta)

    def __getitem__(self, indexer):
        xytheta = np.hstack((self._xy[indexer], self._theta[indexer, None]))
        return self.__class__(xytheta)

    def __len__(self):
        return self.len

    def __mul__(self, other):
        """
        Performs element-wise pose composition.
        """
        if not(len(self) == 1 or len(other) == 1 or
               len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in "
                             "both or a single transformation in either "
                             "object, got {} transformations in first and "
                             "{} transformations in second object.".format(
                                len(self), len(other)))

        otherx = other._xy[:, 0]
        othery = other._xy[:, 1]

        ctheta = np.cos(self._theta)
        stheta = np.sin(self._theta)
        newx = ctheta * otherx - stheta * othery + self._xy[:, 0]
        newy = stheta * otherx + ctheta * othery + self._xy[:, 1]
        xythetanew = np.vstack((newx, newy, self._theta + other._theta)).T

        return self.__class__(xythetanew)

    def __truediv__(self, other):
        """
        Computes relative pose, similar to MATLAB convention
        (x = A \ b for Ax = b). Example: T1 / T2 = T1.inv() * T2
        Derivation: Do it yourself mate, simple algebra
        """
        if not(len(self) == 1 or len(other) == 1 or
               len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in "
                             "both or a single transformation in either "
                             "object, got {} transformations in first and "
                             "{} transformations in second object.".format(
                                len(self), len(other)))

        otherx = other._xy[:, 0]
        othery = other._xy[:, 1]

        cthetainv = np.cos(-self._theta)
        sthetainv = np.sin(-self._theta)
        newx = cthetainv * (otherx - self._xy[:, 0]) - \
            sthetainv * (othery - self._xy[:, 1])
        newy = sthetainv * (otherx - self._xy[:, 0]) + \
            cthetainv * (othery - self._xy[:, 1])
        xythetanew = np.vstack((newx, newy, other._theta - self._theta)).T

        return self.__class__(xythetanew)

    def inv(self):
        """
        Inverts transformation. Derivation: Use definition of inverse
        followed by basic matrix algebra.
        """
        x = self._xy[:, 0]
        y = self._xy[:, 1]
        thetainv = - self._theta

        # intermediate calcs for inversion

        cinv = np.cos(thetainv)
        sinv = np.sin(thetainv)

        xythetainv = np.vstack((-cinv * x + sinv * y, - sinv * x -
                                cinv * y, thetainv)).T  # computed inverse

        return self.__class__(xythetainv)

    def to_vec(self):
        return np.squeeze(np.hstack((self._xy,
                                     np.atleast_2d(self._theta).T)))

    def magnitude(self):
        return np.linalg.norm(self._xy, axis=-1), np.abs(self._theta)


class SE3:
    def __init__(self, t, R):
        self._single = False

        if t.ndim not in [1, 2] or t.shape[-1] != 3:
            raise ValueError("Expected `t` to have shape (3,) or (N x 3), "
                             "got {}.".format(t.shape))

        # If a single translation is given, convert it to a 2D 1 x 3 matrix but
        # set self._single to True so that we can return appropriate objects
        # in the `to_...` methods
        if t.shape == (3,):
            t = t[None, :]
            self._single = True
            if not R.single:
                raise ValueError("Different number of translations 1 and "
                                 "rotations {}.".format(len(R)))
        elif len(t) == 1:
            self._single = True
            if not R.single:
                raise ValueError("Different number of translations 1 and "
                                 "rotations {}.".format(len(R)))
        elif R.single:
            raise ValueError("Different number of translations {} and "
                             "rotations 1.".format(len(t)))
        else:
            if len(t) != len(R):
                raise ValueError("Differing number of translations {} "
                                 "and rotations {}".format(len(t),len(R)))
        self._t = t
        self._R = R
        self.len = 1 if R.single else len(R)

    @classmethod
    def from_xyzrpy(cls, xyzrpy):
        t = xyzrpy[..., :3]
        R = Rotation.from_euler('xyz', xyzrpy[..., 3:])
        return cls(t, R)

    @classmethod
    def from_xyzquat(cls, t, quat):
        R = Rotation.from_quat(quat)
        return cls(t, R)

    @classmethod
    def from_mat(cls, T):
        R = Rotation.from_quat(T[:, :3, :3])
        t = T[:, :3, 3]
        return cls(t, R)

    def __getitem__(self, indexer):
        return self.__class__(self.t()[indexer], self.R()[indexer])

    def __len__(self):
        return self.len

    def __mul__(self, other):
        """
        Performs element-wise pose composition.
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        return self.__class__(self.R().apply(other.t()) + self.t(), self.R() * other.R())

    def __truediv__(self, other):
        """
        Computes relative pose, similar to MATLAB convention (x = A \ b for Ax = b). Example:
        T1 / T2 = T1.inv() * T2
        TO DO: Broadcasting
        """
        if not(len(self) == 1 or len(other) == 1 or len(self) == len(other)):
            raise ValueError("Expected equal number of transformations in both "
                             "or a single transformation in either object, "
                             "got {} transformations in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        R1_inv = self.R().inv()
        t_new = R1_inv.apply(other.t() - self.t())
        return self.__class__(t_new, R1_inv * other.R())

    def t(self):
        return self._t[0] if self._single else self._t

    def R(self):
        return self._R

    def inv(self):
        R_inv = self.R().inv()
        t_new = -R_inv.apply(self.t())
        return SE3(t_new, R_inv)

    def components(self):
        return self.t(), self.R()

    def to_xyzrpy(self):
        return np.concatenate((self.t(), np.squeeze(self.R().as_euler('xyz'))), axis=-1)

    def to_xyzypr(self):
        return np.concatenate((self.t(), np.squeeze(self.R().as_euler('zyx'))), axis=-1)

    def magnitude(self):
        return np.linalg.norm(self.t(), axis=-1), self.R().magnitude()


def averageSE3(poses, weights=None):
    if weights is None:
        weights = np.ones(len(poses))
    assert np.all(weights >= 0)
    avg_t = np.average(poses.t(), weights=weights, axis=0)
    avg_R = poses.R().mean(weights)
    return SE3(avg_t, avg_R)


def metricSE3(p1, p2, w):
    """
    Computes metric on the cartesian product space representation of SE(3).
    Args:
        p1 (SE3) : set of poses
        p2 (SE3) : set of poses (same size as p1)
        w (float > 0) : weight for attitude component
    """
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in both "
                            "or a single transformation in either object, "
                            "got {} transformations in first and {} transformations in "
                            "second object.".format(
                            len(p1), len(p2)))
    if w < 0:
        raise ValueError("Weight must be non-negative, currently {}".format(w))
    p_rel = p1 / p2
    t_dist = np.linalg.norm(p_rel.t(), axis=-1)
    R_dist = p_rel.R().magnitude()
    return t_dist + w * R_dist


def errorSE3(p1, p2):
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in both "
                            "or a single transformation in either object, "
                            "got {} transformations in first and {} transformations in "
                            "second object.".format(
                            len(p1), len(p2)))
    p_rel = p1 / p2
    return p_rel.magnitude()


def combineSE3(listOfPoses):
    tList = []
    qList = []
    for pose in listOfPoses:
        tList.append(pose.t())
        qList.append(pose.R().as_quat())
    return SE3(np.asarray(tList), Rotation.from_quat(np.asarray(qList)))

def expSE3(twist):
    """
    Applies exponential map to twist vectors
    Args
        twist: N x 6 matrix or 6D vector containing se(3) element(s)
    Returns
        SE3Poses object with equivalent SE3 transforms
    """
    u = twist[..., :3]
    w = twist[..., 3:]

    R = Rotation.from_rotvec(w)
    theta = R.magnitude()
    what = hatOp(w)  # skew symmetric form
    with np.errstate(divide='ignore'):
        B = (1 - np.cos(theta)) / theta ** 2
        C = (theta - np.sin(theta)) / theta ** 3
    if len(twist.shape) == 2:
        B[np.abs(theta) < 1e-3] = 0.5  # limit for theta -> 0
        C[np.abs(theta) < 1e-3] = 1. / 6  # limit for theta -> 0
        V = np.eye(3)[np.newaxis, ...] + B[:, np.newaxis, np.newaxis]\
            * what + C[:, np.newaxis, np.newaxis] * what @ what
        V = V.squeeze()
    else:
        if np.abs(theta) < 1e-3:
            B = 0.5  # limit for theta -> 0
            C = 1. / 6  # limit for theta -> 0
        V = np.eye(3) + B * what + C * what @ what
    t = V @ u[..., np.newaxis]
    return SE3(t.squeeze(), R)


def logSE3(T):
    """
    Applies inverse exponential map to SE3 elements
    Args
        T: SE3Poses element, may have 1 or more (N) poses
    Returns
        Nx6 matrix or 6D vector representing twists
    """
    R = T.R()
    t = T.t()

    theta = R.magnitude()
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta ** 2
        sq_coeff = 1 / theta ** 2 * (1 - A / (2 * B))
    what = hatOp(R.as_rotvec())

    if len(T) > 1:
        A[np.abs(theta) < 1e-3] = 1.  # limit for theta -> 0
        B[np.abs(theta) < 1e-3] = 0.5  # limit for theta -> 0
        sq_coeff[np.abs(theta) < 1e-3] = 1. / 12
        Vinv = np.eye(3)[np.newaxis, ...] - 0.5 *\
            what + sq_coeff[:, np.newaxis, np.newaxis] * what @ what
    else:
        if np.abs(theta) < 1e-3:
            A = 1.  # limit for theta -> 0
            B = 0.5  # limit for theta -> 0
            sq_coeff = 1. / 12
        Vinv = np.eye(3) - 0.5 * what + sq_coeff * what @ what
    u = Vinv @ t[..., np.newaxis]
    return np.concatenate((u.squeeze(), R.as_rotvec()), axis=-1)


def hatOp(vec):
    """
    Turns Nx3 vector into Nx3x3 skew skymmetric representation.
    Works for single 3D vector also.
    """
    if len(vec.shape) == 2:
        mat = np.zeros((vec.shape[0], 3, 3))
    else:
        mat = np.zeros((3, 3))
    mat[..., 1, 0] = vec[..., 2]
    mat[..., 2, 1] = vec[..., 0]
    mat[..., 0, 2] = vec[..., 1]
    if len(vec.shape) == 2:
        mat = mat - np.transpose(mat, (0, 2, 1))
    else:
        mat = mat - mat.transpose()
    return mat


def wrapAngle(angles):
    """
    Wrap set of angles to [-pi, pi)
    """
    angles = (angles + np.pi) % (2 * np.pi)
    angles = np.atleast_1d(angles)
    angles[angles < 0.] += 2. * np.pi
    return np.squeeze(angles - np.pi)
