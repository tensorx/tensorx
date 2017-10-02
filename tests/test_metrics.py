import unittest
import tensorflow as tf
import tensorx.metrics as metrics
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


if __name__ == '__main__':
    unittest.main()
