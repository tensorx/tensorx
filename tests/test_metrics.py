import unittest
import tensorflow as tf
from tensorx import transform, metrics
import numpy as np
from numpy.linalg import norm
from tensorx import test_utils

import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def np_cosine_dist(u, v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


class MyTestCase(test_utils.TestCase):

    def test_cosine_similarity(self):
        v1_data = np.random.normal(-1., 1., [100])
        v2_data = np.random.normal(-1., 1., [100])
        v3_data = np.random.normal(-1., 1., [100])

        v1 = tf.constant(v1_data, dtype=tf.float32)
        v2 = tf.constant(v2_data, dtype=tf.float32)

        dist0 = metrics.cosine_distance(v1, v1)

        v1s = transform.to_sparse(v1)
        dist1_dense = metrics.cosine_distance(v1, v2)
        dist1_sparse = metrics.sparse_cosine_distance(v1s, v2)

        v3 = tf.constant(v3_data, dtype=tf.float32)

        dist2_dense = metrics.cosine_distance(v1, v3)
        dist2_sparse = metrics.sparse_cosine_distance(v1s, v3)

        with self.cached_session(use_gpu=True):
            self.assertAllClose(dist0, 0, atol=1e-1)
            self.assertIsInstance(v1s, tf.SparseTensor)
            self.assertEqual(dist1_dense, dist1_sparse)
            self.assertEqual(dist2_dense, dist2_sparse)

    def test_sparse_cosine_distance(self):
        dense_shape1 = [10, 100]
        dense_shape2 = [1000, 10000]

        data1 = np.random.normal(0., 1., dense_shape1)
        v1 = tf.constant(data1, dtype=tf.float32)

        v4s = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]],
                              values=[1., 2., 1.],
                              dense_shape=dense_shape2)
        v4 = tf.sparse.to_dense(v4s)

        dist1 = metrics.cosine_distance(v1, v1)
        with self.cached_session(use_gpu=True):
            self.assertAllClose(dist1, 0., atol=1e-6)

            start = time.time()
            self.eval(metrics.cosine_distance(v4, v4))
            end = time.time()
            time1 = end - start

            start = time.time()
            self.eval(metrics.sparse_cosine_distance(v4s, v4))
            end = time.time()
            time2 = end - start

            # self.assertGreater(time1, time2)

    def test_broadcast_cosine_distance(self):
        data1 = [[1., 0., 1.], [1., 0., 1.]]
        data2 = [[-1., 0., -1], [-1., 0., -1], [-1., 0., -1]]

        distance = metrics.batch_cosine_distance(data1, data2)
        distance_shape = tf.shape(distance)
        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(distance_shape, [2, 3])

    def test_broadcast_sparse_cosine_distance(self):
        data1 = [[1., 0., 1.], [1., 0., 1.]]
        data2 = [[-1., 0., -1], [-1., 0., -1], [-1., 0., -1]]

        data1 = tf.constant(data1)
        data2 = tf.constant(data2)
        data1_sp = transform.to_sparse(data1)
        data2_sp = transform.to_sparse(data2)

        dist_sp_dense12 = metrics.batch_sparse_cosine_distance(data1_sp, data2)
        dist_sp_dense12_shape = tf.shape(dist_sp_dense12)
        dist_dense12 = metrics.batch_cosine_distance(data1, data2)

        dist_sp_dense1 = metrics.batch_sparse_cosine_distance(data1_sp, data1)
        dist_sp_dense2 = metrics.batch_sparse_cosine_distance(data2_sp, data2)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(dist_sp_dense12_shape, [2, 3])
            self.assertArrayEqual(dist_sp_dense12, dist_dense12)

            self.assertAllClose(dist_sp_dense1, 0, atol=1e-6)
            self.assertAllClose(dist_sp_dense2, 0, atol=1e-6)

    def test_torus_l1_distance(self):
        points1 = [1]
        points2 = [[1], [3]]

        dist1 = metrics.torus_l1_distance(points1, [4])
        dist1_rank = tf.rank(dist1)

        dist2 = metrics.torus_l1_distance(points2, [4])
        dist2_rank = tf.rank(dist2)

        with self.cached_session(use_gpu=True):
            self.assertEqual(dist1_rank, 1)
            self.assertEqual(dist2_rank, 2)

    def test_batch_manhattan_dist(self):
        tensor1 = transform.to_sparse([[1., 0., 1.], [1., 0., 1.]])
        tensor2 = transform.to_sparse([[0., 0., -1.], [0., 0., -1.], [-1., 0., -1.]])
        expected = [[3., 3., 4.], [3., 3., 4.]]

        dist = metrics.batch_manhattan_distance(tensor1, tensor2)
        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(dist, expected)


if __name__ == '__main__':
    test_utils.main()
