"""Tests for tensor transformations module"""
import unittest

import tensorflow as tf
import numpy as np

import tensorx.transform as transform


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

        result = transform.sparse_put(tensor, sp_values)
        result = tf.sparse_tensor_to_dense(result)

        np.testing.assert_array_equal(expected.eval(), result.eval())

    def test_dense_put(self):
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[3, 1], [1, 1]])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])
        result = transform.dense_put(tensor, sp_values)

        np.testing.assert_array_equal(expected.eval(), result.eval())

        # updates are cast to the given tensor type
        # 0 as update should also work since sparse tensors have a set of indices used to
        # make the updated indices explicit
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[0, 1], [1, 1]])
        sp_values = tf.SparseTensor(indices=[[0, 0]], values=tf.constant([0], dtype=tf.float32), dense_shape=[2, 2])

        result = transform.dense_put(tensor, sp_values)
        np.testing.assert_array_equal(expected.eval(), result.eval())

        # test with unknown input batch dimension
        ph = tf.placeholder(dtype=tf.int32, shape=[None, 2])
        data = np.ones([2, 2])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[0], dense_shape=[2, 2])
        result = transform.dense_put(ph, sp_values)
        np.testing.assert_array_equal(expected.eval(), result.eval({ph: data}))

    def test_enum_row(self):
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        data = [[0, 1, 3], [1, 2, 3]]
        expected = [[0, 0], [0, 1], [0, 3], [1, 1], [1, 2], [1, 3]]

        result = transform.enum_row(ph)
        result = result.eval({ph: data})
        np.testing.assert_array_equal(expected, result)

        ph = tf.placeholder(dtype=tf.int64, shape=[None, 3])
        result = transform.enum_row(ph)
        result = result.eval({ph: data})
        np.testing.assert_array_equal(expected, result)

    def test_fill_sp_ones(self):
        sp_indices = tf.SparseTensorValue(indices=[[0, 0], [1, 0]], values=[], dense_shape=[2, 2])
        sp_placeholder = tf.sparse_placeholder(tf.float32, shape=[None, 2])

        try:
            tf.sparse_tensor_to_dense(sp_placeholder).eval({sp_placeholder: sp_indices})
        except tf.errors.InvalidArgumentError:
            pass

        def default_values(): return transform.fill_sp_ones(sp_placeholder)
        def forward(): return sp_placeholder

        n_values = tf.shape(sp_placeholder.values)[0]
        sp_ones = tf.cond(tf.equal(n_values, 0),
                          true_fn=default_values,
                          false_fn=forward)
        # dense_result = tf.sparse_tensor_to_dense(sp_ones).eval({sp_placeholder: sp_indices})
        # print(dense_result)

    def test_pairs(self):
        tensor1 = [[0], [1]]
        tensor2 = [1, 2]

        expected = [[0, 1], [1, 1], [0, 2], [1, 2]]

        result = transform.pairs(tensor1, tensor2)

        np.testing.assert_array_equal(expected, result.eval())

    def test_to_sparse(self):
        c = [[1, 0], [2, 3]]

        sparse_indices, sparse_values = transform.to_sparse(c)

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

    def test_to_sparse_zero(self):
        shape = [2, 3]
        data_zero = np.zeros(shape)
        sp_i, sp_v = transform.to_sparse(data_zero)

        self.assertEqual(sp_i.eval().indices.shape[0], 0)
        self.assertEqual(sp_v.eval().indices.shape[0], 0)

        dense = tf.sparse_tensor_to_dense(sp_i)
        np.testing.assert_array_equal(dense.eval(), np.zeros(shape))

    def test_empty_sparse_tensor(self):
        dense_shape = [2, 2]
        empty = transform.empty_sparse_tensor(dense_shape)
        dense_empty = tf.sparse_tensor_to_dense(empty)
        np.testing.assert_array_equal(dense_empty.eval(), np.zeros(dense_shape))


if __name__ == '__main__':
    unittest.main()
