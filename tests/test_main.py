import unittest
import numpy as np
from numpy.testing import assert_allclose

import pybind11_package_example as mod
from pybind11_package_example import _cpp

class MainTest(unittest.TestCase):
    def test_matrix_vector_product(self):
        M = np.array([[1,2],[3,4.]])
        v = np.array([0.1, 0.2])
        x = mod.matrix_vector_product(M, v)
        ans = np.array([0.5, 1.1])

        assert_allclose(x, ans)

    def test_matrix_vector_product_cpp(self):
        M = np.array([[1,2],[3,4.]], dtype=np.float32)
        v = np.array([0.1, 0.2], dtype=np.float32)
        x = np.zeros(2, dtype=np.float32)
        _cpp.matrix_vector_product(M, v, x)
        ans = np.array([0.5, 1.1])

        assert_allclose(x, ans)
