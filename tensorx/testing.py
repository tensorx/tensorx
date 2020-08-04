import unittest
import numpy as np
from tensorx import Layer
from typing import Any
from numbers import Number
from tensorflow import Tensor, Variable


class TestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second):
        if isinstance(first, Layer):
            first = first()
        if isinstance(second, Layer):
            second = second()

        self.assertTrue(np.array_equal(first, second))

    def assertArrayNotEqual(self, first, second):
        if isinstance(first, Layer):
            first = first()
        if isinstance(second, Layer):
            second = second()

        self.assertFalse(np.array_equal(first, second))

    def assertAlmostEqual(self, first, second, places=None, msg=None,
                          delta=None):
        if isinstance(first, (Layer)):
            first = first()
        if isinstance(second, (Layer)):
            second = second()

        super().assertAlmostEqual(first, second, places, msg, delta)

    def assertAllClose(self, actual, desired, rtol=1e-7, atol=0., verbose=True):
        """ It compares the difference between actual and desired to
        ``atol + rtol * abs(desired)``.

        if actual or desired are Tensors, tries to evaluate them, so this
        should be called inside a ``with self.cashed_session():``

        Args:
            actual: array obtained
            desired: desired array value
            rtol: relative tolerance
            atol: absolute tolerance
            verbose: if True, the conflicting values are appended to the error message

        Raises:
            AssertionError: If actual and desired are not equal up to specified precision.

        """
        if isinstance(actual, Layer):
            actual = actual()
        if isinstance(desired, Layer):
            desired = desired()

        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=verbose)
