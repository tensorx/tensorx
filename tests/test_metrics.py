import unittest
import tensorflow as tf
from tensorx import transform, metrics
import numpy as np
from numpy.linalg import norm

import time


def np_cosine_dist(u, v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_cosine_similarity_rounding(self):
        v1 = [1., 1.]
        dist = metrics.cosine_distance(v1, v1)
        # print("{0:.16f}".format(dist.eval()))

    def test_cosine_similarity(self):
        v1_data = np.random.normal(-1., 1., [100])
        v2_data = np.random.normal(-1., 1., [100])
        v3_data = np.random.normal(-1., 1., [100])

        v1 = tf.constant(v1_data, dtype=tf.float32)
        v2 = tf.constant(v2_data, dtype=tf.float32)

        dist0 = metrics.cosine_distance(v1, v1)
        self.assertTrue(np.allclose(dist0.eval(), 0, atol=1e-6))

        v1s = transform.to_sparse(v1)
        self.assertIsInstance(v1s, tf.SparseTensor)

        dist1_dense = metrics.cosine_distance(v1, v2)
        dist1_sparse = metrics.sparse_cosine_distance(v1s, v2)

        # print(dist1_dense.eval())
        # print(dist1_sparse.eval())
        self.assertEqual(dist1_dense.eval(), dist1_sparse.eval())

        v3 = tf.constant(v3_data, dtype=tf.float32)

        dist2_dense = metrics.cosine_distance(v1, v3)
        dist2_sparse = metrics.sparse_cosine_distance(v1s, v3)
        self.assertEqual(dist2_dense.eval(), dist2_sparse.eval())

    def test_sparse_cosine_distance_1d(self):
        dense_shape = [100]
        data1 = np.random.normal(0., 1., dense_shape)
        v1 = tf.constant(data1, dtype=tf.float32)
        data2 = np.random.normal(0., 1., dense_shape)
        v2 = tf.constant(data2, dtype=tf.float32)

        # we want to compute the cosine distance along dimension 1 because we have shape [1,2]

        dist1 = metrics.cosine_distance(v1, v1)
        v3s = tf.SparseTensor(indices=[[0], [1]], values=[1., 2.], dense_shape=dense_shape)
        v3 = tf.sparse_tensor_to_dense(v3s)

        dist2_dense = metrics.cosine_distance(v3, v2)
        dist2_sparse = metrics.sparse_cosine_distance(v3s, v2)

    def test_sparse_cosine_distance(self):
        dense_shape = [10, 100]

        data1 = np.random.normal(0., 1., dense_shape)
        data2 = np.random.normal(0., 1., dense_shape)
        v1 = tf.constant(data1, dtype=tf.float32)
        v2 = tf.constant(data2, dtype=tf.float32)

        # we want to compute the cosine distance along dimension 1 because we have shape [1,2]

        dist1 = metrics.cosine_distance(v1, v1)
        self.assertTrue(np.allclose(dist1.eval(), 0., atol=1e-6))

        # v3s = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1., 2., 1.], dense_shape=dense_shape)
        # v3 = tf.sparse_tensor_to_dense(v3s)

        # dist2_dense = metrics.cosine_distance(v3, v2)
        # dist2_sparse = metrics.sparse_cosine_distance(v3s, v2)
        # np.testing.assert_array_almost_equal(dist2_dense.eval(), dist2_sparse.eval())

        # dist33 = metrics.sparse_cosine_distance(v3s, v3)
        # print(dist33.eval())
        #
        # self.assertTrue(np.allclose(dist33.eval(), 0., atol=1.e-6))

        dense_shape = [1000, 10000]
        v4s = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1., 2., 1.], dense_shape=dense_shape)
        v4 = tf.sparse_tensor_to_dense(v4s)

        start = time.time()
        metrics.cosine_distance(v4, v4).eval()
        end = time.time()
        time1 = end - start

        start = time.time()
        metrics.sparse_cosine_distance(v4s, v4).eval()
        end = time.time()
        time2 = end - start
        self.assertGreater(time1, time2)

    def test_sparse_cosine_dist(self):
        """ Makes no sense to test cosine distances between tensors with value 0
        because the dot product between zero vectors will return 0
        and 1 - 0 gives a distance of 1, when in reality the similarity is 1
        """
        dense_shape = [3, 4]
        v3s = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1., 2., 1.], dense_shape=dense_shape)

        v3 = tf.sparse_tensor_to_dense(v3s)
        dist33 = metrics.cosine_distance(v3, v3)
        dist33s = metrics.sparse_cosine_distance(v3s, v3)

        # self.assertTrue(np.allclose(dist33.eval(), 0., atol=1.e-6))

    def test_broadcast_cosine_distance(self):
        data1 = [[1., 0., 1.], [1., 0., 1.]]
        data2 = [[-1., 0., -1], [-1., 0., -1], [-1., 0., -1]]

        distance = metrics.batch_cosine_distance(data1, data2)
        self.assertTrue(np.array_equal(np.shape(distance.eval()), [2, 3]))

    def test_broadcast_sparse_cosine_distance(self):
        data1 = [[1., 0., 1.], [1., 0., 1.]]
        data2 = [[-1., 0., -1], [-1., 0., -1], [-1., 0., -1]]

        data1 = tf.constant(data1)
        data2 = tf.constant(data2)
        data1_sp = transform.to_sparse(data1)
        data2_sp = transform.to_sparse(data2)

        distance = metrics.batch_sparse_cosine_distance(data1_sp, data2)
        distance_dense = metrics.batch_cosine_distance(data1, data2)
        self.assertTrue(np.array_equal(np.shape(distance.eval()), [2, 3]))
        self.assertTrue(np.array_equal(distance.eval(), distance_dense.eval()))

        d01 = metrics.batch_sparse_cosine_distance(data1_sp, data1)
        d02 = metrics.batch_sparse_cosine_distance(data2_sp, data2)

        self.assertTrue(np.allclose(d01.eval(), 0., atol=1e-6))
        self.assertTrue(np.allclose(d02.eval(), 0., atol=1e-6))

    def test_torus_l1_distance(self):
        points1 = [1]
        points2 = [[1], [3]]

        dist1 = metrics.torus_l1_distance(points1, [4])
        self.assertEqual(tf.rank(dist1).eval(), 1)
        # print(dist1.eval())

        dist2 = metrics.torus_l1_distance(points2, [4])
        self.assertEqual(tf.rank(dist2).eval(), 2)
        # print(dist2.eval())

        points2d1 = [[0, 1]]
        dist2d1 = metrics.torus_l1_distance(points2d1, [2, 2])
        # print(dist2d1.eval())

    def test_torus_l1_distance_batch(self):
        points = [[0, 1], [1, 0]]
        dist = metrics.torus_l1_distance(points, [2, 2])
        # print(dist.eval())

    def test_batch_manhattan_dist(self):
        tensor1 = transform.to_sparse([[1., 0., 1.], [1., 0., 1.]])
        tensor2 = transform.to_sparse([[0., 0., -1.], [0., 0., -1.], [-1., 0., -1.]])

        dist = metrics.batch_manhattan_distance(tensor1, tensor2)
        print(dist.eval())


if __name__ == '__main__':
    unittest.main()
