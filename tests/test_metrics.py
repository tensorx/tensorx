import unittest
import tensorflow as tf
from tensorx import metrics
from tensorx import transform
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_cosine_distance(self):
        v1 = tf.random_normal([10])
        v2 = tf.random_normal([10])

        dist0 = metrics.cosine_distance(v1, v1, 0)
        self.assertAlmostEqual(dist0.eval(), 0, places=6)

        v1 = transform.to_sparse(v2)
        self.assertIsInstance(v1, tf.SparseTensor)

        dist0_sparse = metrics.cosine_distance(v1, v1, 0)


if __name__ == '__main__':
    unittest.main()
