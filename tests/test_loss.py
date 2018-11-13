from tensorx import test_utils
import tensorflow as tf
from tensorx.loss import *
import warnings
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestLoss(test_utils.TestCase):

    def test_mse(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = [[0.0, 1.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0]]
        result = mse(labels, predicted)

        labels = predicted
        result_0 = mse(labels, predicted)

        with self.cached_session(use_gpu=True):
            self.assertGreater(result, 0)
            self.assertEqual(result_0, 0)

    def test_binary_cross_entropy(self):
        labels = np.array([[1.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0]])
        predicted = np.zeros([2, 4], np.float64)

        result = binary_cross_entropy(labels, predicted)

        with self.cached_session(use_gpu=True):
            result = self.eval(result)
            self.assertTrue(np.all(np.greater(result, 0)))

    def test_categorical_cross_entropy(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = [[0.1, 0.6, 0.2, 0.1],
                     [0.0, 0.0, 0.0, 1.0]]

        result = categorical_cross_entropy(labels, predicted)
        result = tf.reduce_mean(result)

        with self.cached_session(use_gpu=True):
            self.assertGreater(result, 0)

    def test_hinge_loss(self):
        labels = [[1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0]]
        predicted = [[0.0, -1.0, 0.0, -1.0],
                     [0.0, -1.0, 0.0, -1.0]]

        result = binary_hinge(labels, predicted)

        with self.cached_session(use_gpu=True):
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

        random_prediction = tf.get_variable("random_prediction",
                                            shape=[2, embed_dim],
                                            trainable=False,
                                            initializer=tf.initializers.random_uniform())

        init = tf.global_variables_initializer()

        loss = sparse_cnce_loss(label_features=label_features,
                                model_prediction=random_prediction,
                                weights=embedding_weights, num_samples=3,
                                noise_ratio=0.6,
                                corrupt_labels=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self._cached_session.as_default()
            self.eval(train_step)


if __name__ == '__main__':
    test_utils.main()
