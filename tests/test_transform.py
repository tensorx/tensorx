import unittest

import tensorflow as tf
import numpy as np

import tensorx.transform as tx


class TestTransform(unittest.TestCase):
    def test_sparse_put(self):
        tensor = tf.SparseTensor([[0, 0], [1, 0]], [2, 0.2], [2, 2])
        sp_values = tf.SparseTensor([[0, 0], [0, 1]], [3.0, 32.0], [2, 2])

        expected = tf.constant([[3., 32.], [0.2, 0.]])

        result = tx.sparse_put(tensor, sp_values)

        ss = tf.Session()
        expected = ss.run(expected)

        result = tf.sparse_tensor_to_dense(result)
        result = ss.run(result)
        np.testing.assert_array_equal(expected, result)

    def test_dense_put(self):
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[3, 1], [1, 1]])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])

        result = tx.dense_put(tensor, sp_values)

        ss = tf.Session()
        expected = ss.run(expected)
        result = ss.run(result)

        np.testing.assert_array_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
