"""Tests for tensor transformations module"""
import unittest

import tensorflow as tf
import numpy as np

import tensorx.transform as trans


class TestTransform(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_sparse_put(self):
        tensor = tf.SparseTensor([[0, 0], [1, 0]], [2, 0.2], [2, 2])
        sp_values = tf.SparseTensor([[0, 0], [0, 1]], [3.0, 0], [2, 2])

        expected = tf.constant([[3., 0], [0.2, 0.]])

        result = trans.sparse_put(tensor, sp_values)
        result = tf.sparse_tensor_to_dense(result)

        np.testing.assert_array_equal(expected.eval(), result.eval())

    def test_dense_put(self):
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[3, 1], [1, 1]])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])
        result = trans.dense_put(tensor, sp_values)

        np.testing.assert_array_equal(expected.eval(), result.eval())

        # updates are cast to the given tensor type
        # 0 as update should also work since sparse tensors have a set of indices used to
        # make the updated indices explicit
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[0, 1], [1, 1]])
        sp_values = tf.SparseTensor(indices=[[0, 0]], values=tf.constant([0], dtype=tf.float32), dense_shape=[2, 2])

        result = trans.dense_put(tensor, sp_values)
        np.testing.assert_array_equal(expected.eval(), result.eval())

    def test_enum_row(self):
        indices = tf.constant([[0, 1], [1, 2]], dtype=tf.int64)
        result = trans.enum_row(indices)

        # print(result.eval())
        # print(tf.shape(result).eval())

    def test_to_sparse(self):
        c = [[1, 0], [2, 3]]

        sparse_indices, sparse_values = trans.to_sparse(c)

        dense_shape = tf.shape(c, out_type=tf.int64)
        indices = tf.where(tf.not_equal(c, 0))

        flat_values = tf.reshape(c, [-1])
        flat_indices = tf.where(tf.not_equal(flat_values, 0))
        flat_indices = tf.squeeze(flat_indices)
        flat_indices = tf.mod(flat_indices, dense_shape[1])

        values = tf.gather_nd(c, indices)

        np.testing.assert_array_equal(sparse_indices.indices.eval(), indices.eval())
        np.testing.assert_array_equal(sparse_indices.values.eval(), flat_indices.eval())
        np.testing.assert_array_equal(sparse_values.values.eval(), values.eval())


if __name__ == '__main__':
    unittest.main()
