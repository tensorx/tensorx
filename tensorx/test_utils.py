import unittest
import numpy as np
from tensorx import Layer
from typing import Any
from numbers import Number
from tensorflow import Tensor


class TestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second):
        if isinstance(first, Layer):
            first = first.tensor()
        if isinstance(second, Layer):
            second = second.tensor()

        self.assertTrue(np.array_equal(first, second))

    def assertArrayNotEqual(self, first, second):
        if isinstance(first, Layer):
            first = first.tensor()
        if isinstance(second, Layer):
            second = second.tensor()

        self.assertFalse(np.array_equal(first, second))
