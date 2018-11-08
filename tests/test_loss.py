import unittest
import tensorflow as tf
from tensorx.loss import *
import traceback
import warnings
import sys
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
        labels = np.array([[1.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0]])
        predicted = np.zeros([2, 4], np.float64)

        result = binary_cross_entropy(labels, predicted)
        result = result.eval()
        self.assertTrue(np.all(np.greater(result, 0)))

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

    def test_sparse_cnce_loss(self):
        k_dim = 6
        embed_dim = 4

        warnings.filterwarnings("error")

        label_features = tf.SparseTensorValue(indices=[[0, 0], [0, 5], [1, 2], [1, 1]],
                                              values=[1., -1., -1., 1.],
                                              dense_shape=[2, k_dim])

        embedding_weights = tf.get_variable("lookup",
                                            shape=[k_dim, embed_dim],
                                            initializer=tf.initializers.random_uniform())

        random_prediction = tf.random_uniform([2, embed_dim])
        random_prediction = tf.get_variable("random_prediction", trainable=False, initializer=random_prediction)

        loss = sparse_cnce_loss(label_features=label_features,
                                model_prediction=random_prediction,
                                weights=embedding_weights, num_samples=2,
                                noise_ratio=0.6,
                                corrupt_labels=True)

        init = tf.global_variables_initializer()
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

        # init.run()
        # grads = train_step.run()

        # print(grads)


if __name__ == '__main__':
    unittest.main()
