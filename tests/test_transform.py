"""Tests for tensor transformations module"""
import unittest

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import linalg_ops
from tensorflow.python.client import timeline

import tensorx.transform as transform


class TestTransform(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_repeat(self):
        n = 23
        x = np.array([[1, 2], [3, 4]])
        r = transform.repeat(x, n)

        shape = tf.shape(r).eval()

        self.assertEqual(shape[-1], np.shape(x)[-1] * n)

        x = [[[1], [2]], [[3], [4]]]
        r = transform.repeat(x, n)

        shape = tf.shape(r).eval()

        self.assertEqual(shape[-1], np.shape(x)[-1] * n)

    def test_dropout(self):
        n = 10000
        b = 10
        x = np.ones([b,n])
        keep_prob = 0.5

        drop_x = transform.dropout(x, keep_prob=keep_prob, scale=True)
        expected_avg = np.mean(x)

        self.assertTrue(np.allclose(np.mean(drop_x.eval()),expected_avg,atol=1e-2))

        drop_x = transform.dropout(x, keep_prob=keep_prob, scale=False)
        expected_avg = np.mean(x) * keep_prob

        self.assertTrue(np.allclose(np.mean(drop_x.eval()), expected_avg, atol=1e-2))

    def test_empty_sparse_tensor(self):
        dense_shape = [2, 2]
        empty = transform.empty_sparse_tensor(dense_shape)
        dense_empty = tf.sparse_tensor_to_dense(empty)
        zeros = tf.zeros(dense_shape)
        np.testing.assert_array_equal(dense_empty.eval(), np.zeros(dense_shape))
        np.testing.assert_array_equal(zeros.eval(), dense_empty.eval())

        dense_shape = [4]
        empty = transform.empty_sparse_tensor(dense_shape)
        dense_empty = tf.sparse_tensor_to_dense(empty)
        zeros = tf.zeros(dense_shape)
        np.testing.assert_array_equal(dense_empty.eval(), np.zeros(dense_shape))
        self.assertTrue(tf.reduce_all(tf.equal(zeros, dense_empty)).eval())

    def test_pairs(self):
        tensor1 = [[0], [1]]
        tensor2 = [1, 2]
        expected = [[0, 1], [1, 1], [0, 2], [1, 2]]

        result = transform.pairs(tensor1, tensor2)
        np.testing.assert_array_equal(expected, result.eval())

    def test_batch_to_matrix_indices(self):
        data = [[0, 1, 3], [1, 2, 3]]
        const_data = tf.constant(data)
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        expected = [[0, 0], [0, 1], [0, 3], [1, 1], [1, 2], [1, 3]]

        result = transform.column_indices_to_matrix_indices(const_data, dtype=tf.int64)
        result = result.eval()
        np.testing.assert_array_equal(expected, result)

        result = transform.column_indices_to_matrix_indices(ph, dtype=tf.int64)
        result = result.eval({ph: data})

        np.testing.assert_array_equal(expected, result)

        ph = tf.placeholder(dtype=tf.int64, shape=[None, 3])
        result = transform.column_indices_to_matrix_indices(ph, dtype=tf.int64)
        result = result.eval({ph: data})
        np.testing.assert_array_equal(expected, result)

    def test_batch_to_matrix_indices_2d(self):
        data = [[1, 2], [3, 4]]
        const_data = tf.constant(data)
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        expected = [[0, 1], [1, 2]]

        result = transform.column_indices_to_matrix_indices(const_data, dtype=tf.int64)

    def test_batch_to_matrix_indices_3d(self):
        data = [[[1], [2]], [[3], [4]]]
        const_data = tf.constant(data)
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        expected = [[[0, 1], [1, 2]], [[0, 3], [1, 4]]]

        result = transform.column_indices_to_matrix_indices(const_data, dtype=tf.int64)

        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        const_data = tf.constant(data)
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        expected = [[[0, 1], [0, 2], [1, 3], [1, 4]], [[0, 5], [0, 6], [1, 7], [1, 8]]]

        result = transform.column_indices_to_matrix_indices(const_data, dtype=tf.int64)

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

    def test_fill_sp_ones(self):
        indices = [[0, 0], [1, 0]]
        dense_shape = [2, 2]
        expected = tf.SparseTensorValue(indices=indices, values=[1, 1], dense_shape=dense_shape)

        fill = transform.sparse_ones(indices, dense_shape, dtype=tf.float32)

        fill_dense = tf.sparse_tensor_to_dense(fill)
        expected_dense = tf.sparse_tensor_to_dense(expected)
        expected_dense = tf.cast(expected_dense, tf.float32)

        np.testing.assert_array_equal(fill_dense.eval(), expected_dense.eval())

    def test_to_sparse(self):
        c = [[1, 0], [2, 3]]

        sparse_tensor = transform.to_sparse(c)

        dense_shape = tf.shape(c, out_type=tf.int64)
        indices = tf.where(tf.not_equal(c, 0))

        flat_values = tf.reshape(c, [-1])
        flat_indices = tf.where(tf.not_equal(flat_values, 0))
        flat_indices = tf.squeeze(flat_indices)
        flat_indices = tf.mod(flat_indices, dense_shape[1])

        values = tf.gather_nd(c, indices)

        sp_indices = transform.sparse_indices(sparse_tensor)
        np.testing.assert_array_equal(sparse_tensor.indices.eval(), indices.eval())

        np.testing.assert_array_equal(sp_indices.values.eval(), flat_indices.eval())
        np.testing.assert_array_equal(sparse_tensor.values.eval(), values.eval())

    def test_to_sparse_zero(self):
        shape = [2, 3]
        data_zero = np.zeros(shape)
        sparse_tensor = transform.to_sparse(data_zero)

        self.assertEqual(sparse_tensor.eval().indices.shape[0], 0)

        dense = tf.sparse_tensor_to_dense(sparse_tensor)
        np.testing.assert_array_equal(dense.eval(), np.zeros(shape))

    def test_profile_one_hot_conversions(self):
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        data = [[0, 1, 3],
                [1, 2, 3]]

        dense_shape = [2, 4]

        expected_dense = [[1, 1, 0, 1],
                          [0, 1, 1, 1]]

        sp_one_hot = transform.sparse_one_hot(ph, dense_shape[1])
        dense_one_hot = transform.dense_one_hot(ph, dense_shape[1])

        dense1 = self.ss.run(tf.sparse_tensor_to_dense(sp_one_hot), feed_dict={ph: data})
        dense2 = self.ss.run(dense_one_hot, feed_dict={ph: data})

        np.testing.assert_array_equal(dense1, expected_dense)
        np.testing.assert_array_equal(dense1, dense2)

    def test_indices(self):
        shape = [4]
        xs = transform.grid(shape)
        self.assertTrue(np.ndim(xs.eval()), 1)
        self.assertTrue(np.array_equal(tf.shape(xs).eval(), [4, 1]))

        shape = [4, 4]
        xys = transform.grid(shape)
        self.assertTrue(np.ndim(xys.eval()), 2)
        self.assertTrue(np.array_equal(tf.shape(xys).eval(), [shape[0] * shape[1], 2]))

        shape = [1, 4]
        xys = transform.grid(shape)
        self.assertTrue(np.ndim(xys.eval()), 2)
        self.assertTrue(np.array_equal(tf.shape(xys).eval(), [shape[0] * shape[1], 2]))

    def test_sparse_overlapping(self):
        tensor1 = tf.SparseTensor([[1, 0]], [1], [2, 3])
        tensor2 = tf.SparseTensor([[0, 0], [1, 0], [1, 1]], [3, 4, 5], [2, 3])

        overlap = transform.sparse_overlap(tensor1, tensor2)
        expected_overlap_value = [1]
        self.assertTrue(np.array_equal(expected_overlap_value, overlap.values.eval()))

        overlap2 = transform.sparse_overlap(tensor2, tensor1)
        self.assertTrue(np.array_equal(overlap.indices.eval(), overlap2.indices.eval()))

        # sparse overlapping with no overlapping
        tensor3 = tf.SparseTensor([[0, 1], [1, 1]], [3, 4], [2, 3])
        overlap = transform.sparse_overlap(tensor1, tensor3)
        expected_overlap_value = []
        self.assertTrue(np.array_equal(expected_overlap_value, overlap.values.eval()))

        overlap = transform.sparse_overlap(tensor2, tensor3)
        expected_overlap_value = [5]
        self.assertTrue(np.array_equal(expected_overlap_value, overlap.values.eval()))

    def test_gather_sparse(self):
        tf.reset_default_graph()
        # sess = tf.Session()

        # with tf.name_scope("test_setup"):
        n_runs = 10

        v = np.array([[1, 0, 1], [0, 0, 2], [3, 0, 3]], dtype=np.float32)
        sp = transform.to_sparse(v)

        indices = np.array([[0, 1], [0, 0], [1, 2]], dtype=np.int64)

        gather_sp_tx = transform.gather_sparse(sp, indices)



        with tf.Session() as ss:
            # debug gather sparse with timeline
            # https://www.tensorflow.org/programmers_guide/graph_viz
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary_writer = tf.summary.FileWriter('/tmp/', ss.graph)

            for i in range(n_runs):
                step = i + 1
                ss.run(gather_sp_tx, options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, 'step%d' % step)

                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)

        summary_writer.add_graph(ss.graph)
        summary_writer.close()

        ctf = tl.generate_chrome_trace_format()
        with open('/tmp/timeline.json', 'w') as f:
            f.write(ctf)

    def test_gather_sparse_v2(self):
        sess = tf.Session()

        # dummy data =====================================================================
        num_cols = 100
        num_rows = 10000

        num_gather = 400

        v = np.ones([num_rows * num_cols])
        mask = np.random.choice(int(num_cols * num_rows), int(num_cols * num_rows / 0.8))
        v[mask] = 0
        v = np.reshape(v, [num_rows, num_cols])
        indices = np.random.randint(0, num_rows - 1, [num_gather])
        # ================================================================================

        sp_tensor = transform.to_sparse(v)
        gather = transform.gather_sparse(sp_tensor, indices)

        # measure runtime performance
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        log_writer = tf.summary.FileWriter('/tmp/', sess.graph)

        # for i in range(100):
        i = 1
        stv1 = sess.run(gather, options=run_options, run_metadata=run_metadata)
        log_writer.add_run_metadata(run_metadata, 'test run {}'.format(i))

        log_writer.add_graph(sess.graph)
        log_writer.close()


if __name__ == '__main__':
    unittest.main()
