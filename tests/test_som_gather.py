import unittest
import tensorflow as tf
from tensorx.metrics import cosine_distance
import functools
import numpy as np


def _safe_div(numerator, denominator, name="value"):
    """Computes a safe divide which returns 0 if the denominator is zero.
    Note that the function contains an additional conditional check that is
    necessary for avoiding situations where the loss is zero causing NaNs to
    creep into the gradient computation.
    Args:
      numerator: An arbitrary `Tensor`.
      denominator: `Tensor` whose shape matches `numerator` and whose values are
        assumed to be non-negative.
      name: An optional name for the returned op.
    Returns:
      The element-wise value of the numerator divided by the denominator.
    """
    return tf.where(
        tf.greater(denominator, 0),
        tf.div(numerator, tf.where(
            tf.equal(denominator, 0),
            tf.ones_like(denominator), denominator)),
        tf.zeros_like(numerator),
        name=name)


def gaussian(x, sigma=0.5):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    gauss = tf.exp(_safe_div(-tf.pow(x, 2), tf.pow(sigma, 0.1)))
    return gauss


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_sate_gauss_neighbourhood(self):
        def l1_dist_wrap(center, points):
            center = tf.convert_to_tensor(center)
            other = tf.convert_to_tensor(points)

            size = other.get_shape()
            return tf.minimum(tf.abs(center - other), tf.mod(-(tf.abs(center - other)), size))

        result = gaussian(0., 0.5).eval()
        self.assertEqual(result, 1.)
        result = gaussian([-2., -1., 0., 1., 2.], 2.).eval()
        self.assertTrue(np.array_equal(result[0:2], result[3:5][::-1]))

        print(l1_dist_wrap(3, [0, 1, 2, 3, 4]).eval())

    def test_stage_implementation(self):
        n_inputs = 2
        n_partitions = 3
        n_hidden = 1
        batch_size = 1

        feature_shape = [n_partitions, n_inputs, n_hidden]

        inputs = tf.placeholder(tf.float32, [None, n_inputs], "x")
        som_w = tf.get_variable("p", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))
        feature_w = tf.get_variable("w", feature_shape, tf.float32, tf.random_uniform_initializer(-1., 1.))

        # bmu = tf.reduce_sum(tf.sqrt(som_w-inputs),axis=1)
        # bmu = tf.reduce_sum(tf.sqrt(som_w-inputs),axis=1)
        # bmu_dist = tf.sqrt(tf.reduce_sum(tf.pow((som_w - inputs), 2), 1))
        # som_distances = tf.sqrt(tf.reduce_sum(tf.pow((som_w - inputs), 2), 1))

        som_dist = cosine_distance(inputs, som_w)
        bmu = tf.argmin(som_dist, axis=0)
        # bmu_slice = feature_w[bmu]

        # dummy dense example
        indices = [bmu]
        # indices = tf.constant([0, 2])
        feature_w_slice = tf.gather(feature_w, indices=indices)

        # ext_inputs = tf.expand_dims(inputs,0)
        # features = tf.squeeze(feature_w_slice, [-1])
        features = feature_w_slice

        h = tf.tensordot(inputs, features, axes=[[0, 1], [1, 2]])

        # didn't know but calling global variable int must be done after adding the vars to the graph
        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)

            feed_dict = {inputs: [[1., -1.]]}

            print("som vars:\n", som_w.eval())
            print("feature vars:\n", feature_w.eval())
            # print("dist: \n", som_dist.eval(feed_dict))
            # print(bmu.eval(feed_dict))

            # print("slices:\n", feature_w_slice.eval(feed_dict))
            print("features:\n", features.eval(feed_dict))
            # hidden
            print("result:\n", h.eval(feed_dict))

            print(gaussian(0., 0.5).eval())


if __name__ == '__main__':
    unittest.main()
