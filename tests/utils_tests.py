import unittest
from topometricloc import utils

import numpy as np
import scipy.sparse as sparse


class TestSparseUtils(unittest.TestCase):
    def setUp(self):
        N = 25
        W = 5
        self.mat = sparse.random(N, N, density=0.3, format='csr', dtype=np.float64)
        self.mat.data -= 25. * self.mat.data.max()
        self.vec = np.random.randn(N)

    def test_max_nonzero(self):
        from topometricloc.utils import max_nonzero
        # along both axes
        max_val_nz = max_nonzero(self.mat)
        self.assertAlmostEqual(max_val_nz, self.mat.data.max())
        max_val_nz_0 = max_nonzero(self.mat, axis=0)
        # max along axes 0
        max_0_true = []
        for j in range(self.mat.shape[1]):
            max_along_row = -np.inf
            for i in range(self.mat.shape[0]):
                if self.mat[i, j] != 0. and self.mat[i, j] >= max_along_row:
                    max_along_row = self.mat[i, j]
            max_0_true.append(max_along_row)
        max_0_true = np.array(max_0_true)
        self.assertTrue(np.allclose(max_val_nz_0.toarray().squeeze(), max_0_true))
        # max along axes 1
        max_val_nz_1 = max_nonzero(self.mat, axis=1)
        max_1_true = []
        for i in range(self.mat.shape[0]):
            max_along_row = -np.inf
            for j in range(self.mat.shape[1]):
                if self.mat[i, j] != 0. and self.mat[i, j] >= max_along_row:
                    max_along_row = self.mat[i, j]
            max_1_true.append(max_along_row)
        max_1_true = np.array(max_1_true)
        self.assertTrue(np.allclose(max_val_nz_1.toarray().squeeze(), max_1_true))
        # check it works if an element is > 1
        self.mat.data += 2.
        max_val_nz = max_nonzero(self.mat)
        self.assertAlmostEqual(max_val_nz, self.mat.data.max())

    def test_logsumexp_nonzero(self):
        from topometricloc.utils import logsumexp_nonzero
        from scipy.special import logsumexp
        # along both axes
        lse = logsumexp_nonzero(self.mat)
        lse_true = logsumexp(self.mat.data)
        self.assertAlmostEqual(lse, lse_true)
        # LSE along axes 0
        lse_0 = logsumexp_nonzero(self.mat, axis=0)
        self.mat = self.mat.tocsc()
        lse_0_true = []
        for i in range(self.mat.shape[0]):
            indptr = self.mat.indptr
            res = logsumexp(self.mat.data[indptr[i]:indptr[i+1]])
            lse_0_true.append(res)
        lse_0_true = np.array(lse_0_true)
        np.testing.assert_allclose(lse_0, lse_0_true, atol=1e-25, rtol=0)
        # LSE along axes 1
        lse_1 = logsumexp_nonzero(self.mat, axis=1)
        self.mat = self.mat.tocsr()
        lse_1_true = []
        for i in range(self.mat.shape[1]):
            indptr = self.mat.indptr
            res = logsumexp(self.mat.data[indptr[i]:indptr[i+1]])
            lse_1_true.append(res)
        lse_1_true = np.array(lse_1_true)
        np.testing.assert_allclose(lse_1, lse_1_true, atol=1e-25, rtol=0)

    def test_sparse_nz_sum(self):
        from topometricloc.utils import sparse_nz_sum
        # csr representation, assumes vec is a column vector
        col_sum = sparse_nz_sum(self.mat, self.vec)
        col_sum_true = self.mat.copy()
        for i in range(self.mat.shape[0]):
            indptr = col_sum_true.indptr
            col_sum_true.data[indptr[i]:indptr[i+1]] += self.vec[i]
        np.testing.assert_allclose(col_sum_true.toarray(), col_sum.toarray(), atol=1e-25, rtol=0)
        # change to csc representation, assumes vec is a row vector
        self.mat = self.mat.tocsc()
        row_sum = sparse_nz_sum(self.mat, self.vec)
        row_sum_true = self.mat.copy()
        for i in range(self.mat.shape[0]):
            indptr = row_sum_true.indptr
            row_sum_true.data[indptr[i]:indptr[i+1]] += self.vec[i]
        np.testing.assert_allclose(row_sum_true.toarray(), row_sum.toarray(), atol=1e-25, rtol=0)


if __name__ == '__main__':
    unittest.main()
