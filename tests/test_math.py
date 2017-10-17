import unittest
import tensorflow as tf
import numpy as np

from tensorx import math as mathx
from tensorx import transform


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_l2_norm(self):
        v1 = tf.constant([[2., 0., -1.], [2., 0., -1.]])
        v1s = transform.to_sparse(v1)

        norm1 = tf.norm(v1, axis=1)
        norm1s = mathx.sparse_l2_norm(v1s, axis=1)

        self.assertTrue(np.array_equal(norm1.eval(), norm1s.eval()))


if __name__ == '__main__':
    unittest.main()
