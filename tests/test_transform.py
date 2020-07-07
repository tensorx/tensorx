# suppressing messages only works if set before tensorflow is imported
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorx as tx
import numpy as np

import unittest
from tensorx.testing import TestCase


class TestLayers(TestCase):
    def test_dropout(self):
        n = 2000
        b = 10
        x = tf.ones([b, n])
        prob = 0.5

        drop_x = tx.dropout(x, probability=prob, scale=True)

        actual_avg = tf.reduce_mean(drop_x)
        expected_avg = tf.reduce_mean(x)

        # self.assertAlmostEqual(actual_avg, expected_avg, delta=1e-2)

    def test_dropout_random_tensor(self):
        n = 2000
        b = 4
        x = tf.ones([b, n])
        keep_prob = 0.5

        mask = np.random.uniform(size=x.shape)

        drop_x = tx.dropout(x, probability=keep_prob, random_mask=mask, scale=True)

        actual_avg = tf.reduce_mean(drop_x)
        expected_avg = tf.reduce_mean(x)

        self.assertAlmostEqual(actual_avg, expected_avg, delta=1e-1)

    def test_dropout_unscaled(self):
        n = 2000
        b = 10
        x = tf.ones([b, n])
        keep_prob = 0.5

        drop_x = tx.dropout(x, probability=keep_prob, scale=False)

        actual_avg = tf.reduce_mean(drop_x)
        expected_avg = tf.reduce_mean(x) * keep_prob

        self.assertAlmostEqual(actual_avg, expected_avg, delta=1e-2)

    def test_empty_sparse_tensor(self):
        dense_shape = [2, 2]
        empty = tx.empty_sparse_tensor(dense_shape)
        dense_empty = tf.sparse.to_dense(empty)
        zeros = tf.zeros(dense_shape)
        all_zero = tf.reduce_all(tf.equal(zeros, dense_empty))

        self.assertTrue(all_zero)


if __name__ == '__main__':
    unittest.main()
