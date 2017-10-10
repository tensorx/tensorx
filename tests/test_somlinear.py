import unittest
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import feature_column

from tensorx.metrics import cosine_distance
import functools
import numpy as np

from tensorx.utils import to_tensor_cast
from tensorx import transform


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


def l1_dist_wrap(center, points):
    center = to_tensor_cast(center, tf.float32)
    other = to_tensor_cast(points, tf.float32)

    size = other.get_shape()[-1].value
    return tf.minimum(tf.abs(center - other), tf.mod(-(tf.abs(center - other)), size))


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_sate_gauss_neighbourhood(self):
        result = gaussian(0., 0.5).eval()
        self.assertEqual(result, 1.)
        result = gaussian([-2., -1., 0., 1., 2.], 2.).eval()
        self.assertTrue(np.array_equal(result[0:2], result[3:5][::-1]))

        print(l1_dist_wrap(3, [0, 1, 2, 3, 4]).eval())

    def test_stage_implementation(self):
        n_inputs = 3
        n_partitions = 2
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

        map_indices = tf.range(0, n_partitions, 1)
        neigh_distances = tf.to_float(l1_dist_wrap(bmu, map_indices))
        weights = gaussian(neigh_distances, sigma=0.1)
        neigh_threshold = 1e-6
        neigh_active = tf.greater(weights, neigh_threshold)

        neigh_indices = tf.where(neigh_active)
        neigh_weights = tf.boolean_mask(weights, neigh_active)

        # indices = tf.constant([0, 2])
        w = tf.gather_nd(feature_w, indices=neigh_indices)

        # reshape neigh weights to multiply by 2
        ones = tf.fill(tf.expand_dims(tf.rank(w) - 1, 0), 1)
        bcast_neigh_weights_shape = tf.concat([tf.shape(neigh_weights), ones], 0)
        orig_weights_shape = neigh_weights.get_shape()
        neigh_weights = tf.reshape(neigh_weights, bcast_neigh_weights_shape)

        # Set the weight shape, since after reshaping to bcast_weights_shape,
        # the shape becomes None.
        if w.get_shape().ndims is not None:
            neigh_weights.set_shape(
                orig_weights_shape.concatenate(
                    [1 for _ in range(w.get_shape().ndims - 1)]))

        w2 = w * neigh_weights

        # for dense inputs
        # h = tf.tensordot(inputs, features, axes=[[0, 1], [1, 2]])

        # for sparse inputs
        sp_weights = transform.to_sparse(inputs)
        sp_indices = transform.sp_indices_from_sp_tensor(sp_weights)

        # this doesnt work with rank 3 shapes, we need to expand the sp_indices and sp_values
        # or use map_fn

        def feature_fn(features):
            sparse_h = tf.nn.embedding_lookup_sparse(features, sp_indices, sp_weights=sp_weights, combiner="sum")
            return sparse_h

        sparse_h_map_fn = tf.map_fn(feature_fn, elems=w2, parallel_iterations=8)
        sparse_h_map_fn = tf.reshape(sparse_h_map_fn, [-1, n_hidden])

        # do it without map_fn
        num_indices = tf.shape(sp_weights.values)[0]
        rows = tf.reshape(tf.tile(tf.range(0, n_partitions), [num_indices]), [num_indices, n_partitions])
        rows = tf.transpose(rows)
        rows = tf.reshape(rows, [num_indices * n_partitions, 1])
        rows = tf.cast(rows, tf.int64)

        # tf.tile(tf.constant([[0,1],[0,1]]),multiples=tf.constant([2,1])).eval()
        # indices_tiled = tf.tile(sp_indices.indices, multiples=[n_partitions, 1])
        # indices_tiled = tf.concat([rows,indices_tiled],axis=1)


        flat_indices_tiled = tf.tile(sp_indices.values, multiples=[n_partitions])

        indices_tiled = tf.concat([rows, tf.expand_dims(flat_indices_tiled,axis=1)], axis=1)

        values_tiled = tf.tile(sp_weights.values, multiples=[n_partitions])
        # dense_shape = tf.stack([tf.cast(n_partitions, tf.int64), sp_indices.dense_shape[0], sp_indices.dense_shape[1]],axis=-1)
        dense_shape = tf.stack([tf.cast(n_partitions, tf.int64), sp_indices.dense_shape[1]], axis=-1)

        sp_indices = tf.SparseTensor(indices_tiled, flat_indices_tiled, dense_shape)
        sp_weights = tf.SparseTensor(indices_tiled, values_tiled, dense_shape)

        #sparse_h_no_mapfn = tf.nn.embedding_lookup_sparse(params=tf.squeeze(w2,axis=-1), sp_ids=sp_indices, sp_weights=sp_weights,
         #                                                 combiner="sum", partition_strategy="mod")

        # didn't know but calling global variable int must be done after adding the vars to the graph
        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)

            feed_dict = {inputs: [[1., 0, -1]]}

            # print("neigh distances")
            # print(neigh_distances.eval(feed_dict))

            # print("weights:")
            # print(weights.eval(feed_dict))

            # print("weights:")
            # print(weights.eval(feed_dict))

            # print("neight indices: ")
            # print(neigh_indices.eval(feed_dict))

            # print("neigh weights:")
            # print(neigh_weights.eval(feed_dict))

            # print("neigh gather:")
            # print(w.eval(feed_dict))

            print("neigh gather * weights:")
            print(w2.eval(feed_dict))

            print("neigh gather * weights:")
            print(tf.reshape(w2,shape=[n_partitions*n_inputs,n_hidden]).eval(feed_dict))

            print("rows")
            print(rows.eval(feed_dict))

            print("new_dense_shape")
            print(dense_shape.eval(feed_dict))

            print("indices tiled")
            print(sp_indices.indices.eval(feed_dict))

            print("new indices")
            print(indices_tiled.eval(feed_dict))

            print("sparse h map_fn")
            print(sparse_h_map_fn.eval(feed_dict))

            print("sparse h without map_fn")
            print(sparse_h_no_mapfn.eval(feed_dict))




            # print("som vars:\n", som_w.eval())
            # print("feature vars:\n", feature_w.eval())
            # print("dist: \n", som_dist.eval(feed_dict))
            # print(bmu.eval(feed_dict))

            # print("slices:\n", feature_w_slice.eval(feed_dict))
            # print("features:\n", features.eval(feed_dict))
            # hidden
            # print("result:\n", h.eval(feed_dict))

            # print(gaussian(0., 0.5).eval())


if __name__ == '__main__':
    unittest.main()
