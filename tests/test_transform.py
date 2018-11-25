""" Tests for TensorX transform module
"""
from tensorx import transform, test_utils
from tensorflow.python.framework.sparse_tensor import *
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import dtypes
import os
import numpy as np
from tensorflow import sparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestTransform(test_utils.TestCase):

    def test_sparse_tile(self):
        n = 4
        sp = SparseTensorValue([[0, 0], [0, 1], [1, 2], [2, 3]], [1, 1, 2, 3], [3, 10])
        tilled_sp = transform.sparse_tile(sp, n)

        with self.cached_session(use_gpu=True):
            shape = self.eval(array_ops.shape(tilled_sp))
            self.assertEqual(shape[0], sp.dense_shape[0] * n)
            self.assertEqual(shape[-1], sp.dense_shape[-1])

    def test_repeat(self):
        n = 23
        x = array_ops.constant([[1, 2], [3, 4]])
        r = transform.repeat(x, n)
        dim_r = array_ops.shape(r)[-1]
        dim_x = array_ops.shape(x)[-1]

        with self.cached_session(use_gpu=True):
            dr, dx = self.eval([dim_r, dim_x])
            self.assertEqual(dr, dx * n)

        x = array_ops.constant([[[1], [2]], [[3], [4]]])
        r = transform.repeat(x, n)

        dim_r = array_ops.shape(r)[-1]
        dim_x = array_ops.shape(x)[-1]

        with self.cached_session(use_gpu=True):
            dr, dx = self.eval([dim_r, dim_x])
            self.assertEqual(dr, dx * n)

    def test_dropout(self):
        n = 10000
        b = 10
        x = array_ops.ones([b, n])
        keep_prob = 0.5

        drop_x = transform.dropout(x, keep_prob=keep_prob, scale=True)

        actual_avg = math_ops.reduce_mean(drop_x)
        expected_avg = math_ops.mean(x, axis=-1)

        with self.cached_session(use_gpu=True):
            self.assertAllClose(actual=actual_avg,
                                desired=expected_avg,
                                atol=1e-2)

    def test_dropout_unscaled(self):
        n = 10000
        b = 10
        x = array_ops.ones([b, n])
        keep_prob = 0.5

        drop_x = transform.dropout(x, keep_prob=keep_prob, scale=False)

        actual_avg = math_ops.reduce_mean(drop_x)
        expected_avg = math_ops.mean(x, axis=-1) * keep_prob

        with self.cached_session(use_gpu=True):
            self.assertAllClose(actual=actual_avg,
                                desired=expected_avg,
                                atol=1e-1)

    def test_empty_sparse_tensor(self):
        dense_shape = [2, 2]
        empty = transform.empty_sparse_tensor(dense_shape)
        dense_empty = sparse.to_dense(empty)
        zeros = array_ops.zeros(dense_shape)
        all_zero = math_ops.reduce_all(math_ops.equal(zeros, dense_empty))

        with self.cached_session(use_gpu=True):
            self.assertTrue(self.eval(all_zero))

    def test_pairs(self):
        tensor1 = [[0], [1]]
        tensor2 = [1, 2]
        expected = [[0, 1], [1, 1], [0, 2], [1, 2]]
        result = transform.pairs(tensor1, tensor2)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result, expected)

    def test_to_matrix_indices_2d(self):
        flat_indices = [[0, 1], [1, 2]]
        matrix_indices = transform.to_matrix_indices_2d(flat_indices)

        with self.cached_session(use_gpu=True) as ss:
            self.eval(matrix_indices)

    def test_to_matrix_indices(self):
        data = [[0, 1, 3], [1, 2, 3]]
        const_data = array_ops.constant(data)
        ph = array_ops.placeholder(dtype=dtypes.int64, shape=[2, 3])
        ph_dy_batch = array_ops.placeholder(dtype=dtypes.int64, shape=[None, 3])
        expected = [[0, 0], [0, 1], [0, 3], [1, 1], [1, 2], [1, 3]]

        const_indices = transform.to_matrix_indices_2d(const_data, dtypes.int64)
        ph_indices = transform.to_matrix_indices_2d(ph, dtypes.int64)
        ph_dy_batch_indices = transform.to_matrix_indices_2d(ph_dy_batch, dtypes.int64)

        with self.cached_session(use_gpu=True):
            ph_indices = self.eval(ph_indices, {ph: data})
            ph_dy_batch_indices = self.eval(ph_dy_batch_indices, {ph_dy_batch: data})
            self.assertArrayEqual(const_indices, expected)
            self.assertArrayEqual(ph_indices, expected)
            self.assertArrayEqual(ph_dy_batch_indices, expected)

    def test_batch_to_matrix_indices_3d(self):
        data1 = array_ops.constant([[[1], [2]], [[3], [4]]])
        data2 = array_ops.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        expected1 = [[[0, 1], [1, 2]], [[0, 3], [1, 4]]]
        expected2 = [[[0, 1], [0, 2], [1, 3], [1, 4]], [[0, 5], [0, 6], [1, 7], [1, 8]]]

        indices1 = transform.to_matrix_indices(data1, dtypes.int64)
        indices2 = transform.to_matrix_indices(data2, dtypes.int64)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(indices1, expected1)
            self.assertArrayEqual(indices2, expected2)

    def test_sparse_put(self):
        tensor = SparseTensor([[0, 0], [1, 0]], [2, 0.2], [2, 2])
        sp_values = SparseTensor([[0, 0], [0, 1]], [3.0, 0], [2, 2])
        expected = array_ops.constant([[3., 0], [0.2, 0.]])

        result = transform.sparse_put(tensor, sp_values)
        result = sparse.to_dense(result)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result, expected)

    def test_dense_put(self):
        x = array_ops.ones([2, 2], dtypes.int32)
        expected = array_ops.constant([[3, 1], [1, 1]])

        sp_values = SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])
        result = transform.dense_put(x, sp_values)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result, expected)

    def test_dense_put_zero(self):
        x = array_ops.ones([2, 2], dtypes.int32)
        expected = array_ops.constant([[0, 1], [1, 1]])
        sp_values = SparseTensor(indices=[[0, 0]], values=[0.], dense_shape=[2, 2])
        result = transform.dense_put(x, sp_values)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result, expected)

    def test_fill_sp_ones(self):
        indices = [[0, 0], [1, 0]]
        dense_shape = [2, 2]
        expected = SparseTensorValue(indices=indices, values=[1, 1], dense_shape=dense_shape)

        fill = transform.sparse_ones(indices, dense_shape, dtypes.float32)
        fill_dense = sparse.to_dense(fill)
        expected_dense = sparse.to_dense(expected)
        expected_dense = math_ops.cast(expected_dense, dtypes.float32)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(fill_dense, expected_dense)

    def test_to_sparse(self):
        c = [[1, 0], [2, 3]]

        sparse_tensor = transform.to_sparse(c)

        dense_shape = array_ops.shape(c, out_type=dtypes.int64)
        indices = array_ops.where(math_ops.not_equal(c, 0))

        flat_values = array_ops.reshape(c, [-1])
        flat_indices = array_ops.where(math_ops.not_equal(flat_values, 0))
        flat_indices = array_ops.squeeze(flat_indices)
        flat_indices = math_ops.mod(flat_indices, dense_shape[1])

        values = array_ops.gather_nd(c, indices)

        sp_indices = transform.sparse_indices(sparse_tensor)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(sparse_tensor.indices, indices)
            self.assertArrayEqual(sp_indices.values, flat_indices)
            self.assertArrayEqual(sparse_tensor.values, values)

    def test_to_sparse_zero(self):
        shape = [2, 3]
        data_zero = array_ops.zeros(shape)
        sparse_tensor = transform.to_sparse(data_zero)
        dense = sparse.to_dense(sparse_tensor)
        num_indices = array_ops.shape(sparse_tensor.indices)[0]

        with self.cached_session(use_gpu=True):
            self.assertEqual(num_indices, 0)
            self.assertArrayEqual(dense, data_zero)

    def test_one_hot_conversions(self):
        x = array_ops.constant([[0, 1, 3],
                                [1, 2, 3]])
        num_cols = 4
        expected_dense = [[1, 1, 0, 1],
                          [0, 1, 1, 1]]

        sp_one_hot = transform.sparse_one_hot(x, num_cols)
        dense1 = sparse.to_dense(sp_one_hot)
        dense2 = transform.dense_one_hot(x, num_cols)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(dense1, expected_dense)
            self.assertArrayEqual(dense1, dense2)

    def test_grid(self):
        shape_1d = [4]
        shape_2d = [4, 4]
        xs = transform.grid(shape_1d)
        xys = transform.grid(shape_2d)
        shape_xys = array_ops.shape(xys)

        with self.cached_session(use_gpu=False):
            self.assertEqual(array_ops.rank(xs), 2)
            self.assertEqual(array_ops.rank(xys), 2)

            self.assertArrayEqual(shape_xys, [shape_2d[0] * shape_2d[1], 2])

    def test_sparse_overlapping(self):
        tensor1 = SparseTensor([[1, 0]], [1], [2, 3])
        tensor2 = SparseTensor([[0, 0], [1, 0], [1, 1]], [3, 4, 5], [2, 3])
        tensor3 = SparseTensor([[0, 1], [1, 1]], [3, 4], [2, 3])

        overlap12 = transform.sparse_overlap(tensor1, tensor2)
        overlap21 = transform.sparse_overlap(tensor2, tensor1)
        overlap23 = transform.sparse_overlap(tensor2, tensor3)
        no_overlap = transform.sparse_overlap(tensor1, tensor3)
        expected_overlap = [1]
        expected_overlap23 = [5]
        expected_no_overlap = []

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(overlap12.values, expected_overlap)
            self.assertArrayEqual(overlap21.indices, overlap12.indices)
            # sparse overlapping with no overlapping
            self.assertArrayEqual(no_overlap.values, expected_no_overlap)
            self.assertArrayEqual(overlap23.values, expected_overlap23)

    def test_gather_sparse(self):
        v = array_ops.constant([[1, 0, 1], [0, 0, 2], [3, 0, 3]], dtypes.float32)
        sp = transform.to_sparse(v)

        indices = np.array([[0, 1], [2, 0]], dtype=np.int64)

        gather_sp = transform.gather_sparse(sp, indices)
        gather = sparse.to_dense(gather_sp)
        expected = [[1, 0, 1],
                    [0, 0, 2],
                    [3, 0, 3],
                    [1, 0, 1]]

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(gather, expected)

    def test_sort_by_first(self):
        v1 = array_ops.constant([[3, 1], [2, 1]])
        sorted1 = [[1, 3], [1, 2]]
        v2 = array_ops.constant([[1, 2], [1, 2]])
        sorted2 = [[2, 1], [2, 1]]

        s1, s2 = transform.sort_by_first(v1, v2, ascending=True)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(s1, sorted1)
            self.assertArrayEqual(s2, sorted2)


if __name__ == '__main__':
    test_utils.main()
