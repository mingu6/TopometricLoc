import unittest
import numpy as np
import numpy.testing as test

from motion_model import shortest_dist_segments


class TestMotionFuncs(unittest.TestCase):
    def testBothOut(self):
        """
        Case 1-4: case where s, t is outside of [0, 1]^2.
            Segment 1: (0, 0, 0) to (1, 1, 0)
            Segment 2: (2, 1, 0) to (2, 2, 0)
            Expected distance is 1
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.],
                       [1., 1., 0.]])

        p1 = np.array([[2., 1., 0.],
                       [2., 1., 0.],
                       [2., 2., 0.],
                       [2., 2., 0.]])

        u = np.array([[ 1.,  1.,  0.],
                      [-1., -1.,  0.],
                      [ 1.,  1.,  0.],
                      [-1., -1.,  0.]])

        v = np.array([[ 0.,  1.,  0.],
                      [ 0.,  1.,  0.],
                      [ 0., -1.,  0.],
                      [ 0., -1.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 1., atol=1e-8)

    def testOneIn(self):
        """
        Case 5-8: case where s is in [0, 1] but t is not.
            Segment 1: (0, 0, 0) to (1, 1, 0)
            Segment 2: (2, 0, 0) to (2, 2, 0)
            Expected distance is 1
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.],
                       [1., 1., 0.]])

        p1 = np.array([[2., 0., 0.],
                       [2., 0., 0.],
                       [2., 2., 0.],
                       [2., 2., 0.]])

        u = np.array([[ 1.,  1.,  0.],
                      [-1., -1.,  0.],
                      [ 1.,  1.,  0.],
                      [-1., -1.,  0.]])

        v = np.array([[ 0.,  2.,  0.],
                      [ 0.,  2.,  0.],
                      [ 0., -2.,  0.],
                      [ 0., -2.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 1.)

    def testRegular(self):
        """
        Case 9-12: case where s, t is in [0, 1]^2
            Segment 1: (0, 0, 0) to (1, 1, 0)
            Segment 2: (0, 1, 1) to (1, 0, 1)
            Expected distance is 1
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.],
                       [1., 1., 0.]])

        p1 = np.array([[0., 1., 1.],
                       [0., 1., 1.],
                       [1., 0., 1.],
                       [1., 0., 1.]])

        u = np.array([[ 1.,  1.,  0.],
                      [-1., -1.,  0.],
                      [ 1.,  1.,  0.],
                      [-1., -1.,  0.]])

        v = np.array([[ 1., -1.,  0.],
                      [ 1., -1.,  0.],
                      [-1.,  1.,  0.],
                      [-1.,  1.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 1., atol=1e-8)

    def testParallelOffset(self):
        """
        Case 13-16: Parallel lines, not touching
            Segment 1: (0, 0, 0) to (1, 1, 0)
            Segment 2: (0, 1, 0) to (1, 2, 0)
            Expected distance is 1
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 1., 0.],
                       [0., 0., 0.],
                       [1., 1., 0.]])

        p1 = np.array([[0., 1., 0.],
                       [0., 1., 0.],
                       [1., 2., 0.],
                       [1., 2., 0.]])

        u = np.array([[ 1.,  1.,  0.],
                      [-1., -1.,  0.],
                      [ 1.,  1.,  0.],
                      [-1., -1.,  0.]])

        v = np.array([[ 1.,  1.,  0.],
                      [ 1.,  1.,  0.],
                      [-1., -1.,  0.],
                      [-1., -1.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 1., atol=1e-8)

    def testTouching(self):
        """
        Case 17-20: touching lines, not parallel
            Segment 1: (0, 0, 0) to (1, 0, 0)
            Segment 2: (1, 0, 0) to (2, 1, 0)
            Expected distance is 0
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 0.]])

        p1 = np.array([[1., 0., 0.],
                       [1., 0., 0.],
                       [2., 1., 0.],
                       [2., 1., 0.]])

        u = np.array([[ 1.,  0.,  0.],
                      [-1.,  0.,  0.],
                      [ 1.,  0.,  0.],
                      [-1.,  0.,  0.]])

        v = np.array([[ 1.,  1.,  0.],
                      [ 1.,  1.,  0.],
                      [-1., -1.,  0.],
                      [-1., -1.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 0., atol=1e-8)

    def testCollinearNoOverlap(self):
        """
        Case 21-24: collinear segments, not touching
            Segment 1: (0, 0, 0) to (1, 0, 0)
            Segment 2: (2, 0, 0) to (3, 0, 0)
            Expected distance is 1
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 0.]])

        p1 = np.array([[2., 0., 0.],
                       [2., 0., 0.],
                       [3., 0., 0.],
                       [3., 0., 0.]])

        u = np.array([[ 1.,  0.,  0.],
                      [-1.,  0.,  0.],
                      [ 1.,  0.,  0.],
                      [-1.,  0.,  0.]])

        v = np.array([[ 1.,  0.,  0.],
                      [ 1.,  0.,  0.],
                      [-1.,  0.,  0.],
                      [-1.,  0.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 1., atol=1e-8)

    def testCollinearOverlap(self):
        """
        Case 25-28: collinear segments, overlapping subsegment
            Segment 1: (0, 0, 0) to (1, 0, 0)
            Segment 2: (0.5, 0, 0) to (2.5, 0, 0)
            Expected distance is 1
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 0.]])

        p1 = np.array([[0.5, 0., 0.],
                       [0.5, 0., 0.],
                       [2.5, 0., 0.],
                       [2.5, 0., 0.]])

        u = np.array([[ 1.,  0.,  0.],
                      [-1.,  0.,  0.],
                      [ 1.,  0.,  0.],
                      [-1.,  0.,  0.]])

        v = np.array([[ 2.,  0.,  0.],
                      [ 2.,  0.,  0.],
                      [-2.,  0.,  0.],
                      [-2.,  0.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 0., atol=1e-8)

    def testCollinearTouching(self):
        """
        Case 29-32: collinear segments, intersection is a single point
            Segment 1: (0, 0, 0) to (1, 0, 0)
            Segment 2: (1, 0, 0) to (4, 0, 0)
            Expected distance is 0
        """
        p0 = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.],
                       [1., 0., 0.]])

        p1 = np.array([[1., 0., 0.],
                       [1., 0., 0.],
                       [4., 0., 0.],
                       [4., 0., 0.]])

        u = np.array([[ 1.,  0.,  0.],
                      [-1.,  0.,  0.],
                      [ 1.,  0.,  0.],
                      [-1.,  0.,  0.]])

        v = np.array([[ 3.,  0.,  0.],
                      [ 3.,  0.,  0.],
                      [-3.,  0.,  0.],
                      [-3.,  0.,  0.]])

        d = shortest_dist_segments(p0, u, p1, v)
        test.assert_allclose(d, 0., atol=1e-8)

if __name__ == '__main__':
    unittest.main()
