import unittest
import tensorflow as tf
from tensorx.loss import mse, binary_cross_entropy, categorical_cross_entropy, binary_hinge
import numpy as np


class TestLoss(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_mse(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = [[0.0, 1.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0]]
        result = mse(labels, predicted)

        self.assertGreater(result.eval(), 0)

        labels = predicted
        result = mse(labels, predicted)
        self.assertEqual(result.eval(), 0)

    def test_binary_cross_entropy(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = np.zeros([2, 4])

        result = binary_cross_entropy(labels, predicted)
        result = result.eval()
        self.assertGreater(result, 0)

    def test_categorical_cross_entropy(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = [[0.1, 0.6, 0.2, 0.1],
                     [0.0, 0.0, 0.0, 1.0]]

        result = categorical_cross_entropy(labels, predicted)
        result = tf.reduce_mean(result)
        result = result.eval()
        self.assertGreater(result, 0)

    def test_hinge_loss(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = [[0.0, -1.0, 0.0, -1.0],
                     [0.0, -1.0, 0.0, -1.0]]

        result = binary_hinge(labels, predicted)
        result = result.eval()
        self.assertGreater(result, 0)


if __name__ == '__main__':
    unittest.main()
