import unittest
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import feature_column

from tensorx.metrics import pairwise_cosine_distance
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
    res = tf.div(numerator, tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)),
    res = tf.where(tf.is_finite(res), res, tf.zeros_like(res))
    return res


def gaussian(x, sigma=0.5):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    sigma = tf.expand_dims(sigma,-1)

    gauss = tf.exp(_safe_div(-tf.pow(x, 2), tf.pow(sigma, 2)))
    gauss = tf.squeeze(gauss,0)
    return gauss


def l1_dist_wrap(centers, points):
    centers = to_tensor_cast(centers, tf.float32)
    other = to_tensor_cast(points, tf.float32)

    size = other.get_shape()[-1].value
    return tf.minimum(tf.abs(centers - other), tf.mod(-(tf.abs(centers - other)), size))


def broadcast_reshape(tensor1, tensor2):
    """
    reshapes the first tensor to be broadcastable to the second

    Args:
        tensor1:
        tensor2:

    Returns:

    """
    # reshape neigh weights to multiply by each of gathered w
    ones = tf.fill(tf.expand_dims(tf.rank(tensor2) - 1, 0), 1)
    bcast_neigh_weights_shape = tf.concat([tf.shape(tensor1), ones], 0)
    orig_weights_shape = tensor1.get_shape()
    result = tf.reshape(tensor1, bcast_neigh_weights_shape)

    # Set the weight shape, since after reshaping to bcast_weights_shape,
    # the shape becomes None.
    if tensor2.get_shape().ndims is not None:
        result.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(tensor2.get_shape().ndims - 1)]))

    return result


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
        bias = tf.get_variable("b", [n_partitions, n_hidden], tf.float32, tf.random_uniform_initializer(-1., 1.))

        # bmu = tf.reduce_sum(tf.sqrt(som_w-inputs),axis=1)
        # bmu = tf.reduce_sum(tf.sqrt(som_w-inputs),axis=1)
        # bmu_dist = tf.sqrt(tf.reduce_sum(tf.pow((som_w - inputs), 2), 1))
        # som_distances = tf.sqrt(tf.reduce_sum(tf.pow((som_w - inputs), 2), 1))

        som_dist = pairwise_cosine_distance(inputs, som_w)
        bmu = tf.argmin(som_dist, axis=1)

        map_indices = tf.range(0, n_partitions, 1)
        neigh_distances = tf.to_float(l1_dist_wrap(bmu, map_indices))
        weights = gaussian(neigh_distances, sigma=0.1)
        neigh_threshold = 1e-6
        neigh_active = tf.greater(weights, neigh_threshold)

        neigh_indices = tf.where(neigh_active)
        neigh_weights = tf.boolean_mask(weights, neigh_active)

        # indices = tf.constant([0, 2])
        w = tf.gather_nd(feature_w, indices=neigh_indices)
        b = tf.gather_nd(bias, indices=neigh_indices)

        # reshape neigh weights to multiply by each of gathered w
        """
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
        """
        b2 = b * neigh_weights
        neigh_weights = broadcast_reshape(neigh_weights, w)

        # bias_weights = broadcast_reshape(neigh_weights,b)

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

        rows_2 = tf.reshape(rows, [n_partitions, num_indices, 1])

        # tf.tile(tf.constant([[0,1],[0,1]]),multiples=tf.constant([2,1])).eval()
        # indices_tiled = tf.tile(sp_indices.indices, multiples=[n_partitions, 1])
        # indices_tiled = tf.concat([rows,indices_tiled],axis=1)

        flat_indices_tiled = tf.tile(sp_indices.values, multiples=[n_partitions])
        indices_tiled = tf.concat([rows, tf.expand_dims(flat_indices_tiled, axis=1)], axis=1)

        flat_indices_2 = tf.reshape(flat_indices_tiled, [n_partitions, num_indices, 1])
        rows_2 = tf.reshape(rows, [n_partitions, num_indices, 1])
        indices_tiled_2 = tf.concat([rows_2, flat_indices_2], axis=2)

        values_tiled = tf.tile(sp_weights.values, multiples=[n_partitions])
        values_tiled_2 = tf.reshape(values_tiled, [n_partitions, num_indices, 1])

        # dense_shape = tf.stack([tf.cast(n_partitions, tf.int64), sp_indices.dense_shape[0], sp_indices.dense_shape[1]],axis=-1)
        dense_shape = tf.stack([tf.cast(num_indices * n_partitions, tf.int64), sp_indices.dense_shape[1]], axis=-1)

        sp_indices = tf.SparseTensor(indices_tiled, flat_indices_tiled, dense_shape)
        sp_weights = tf.SparseTensor(indices_tiled, values_tiled, dense_shape)

        # sparse_h_no_mapfn = tf.nn.embedding_lookup_sparse(params=w2, sp_ids=sp_indices, sp_weights=sp_weights,
        #                                                 combiner="sum")
        sparse_h_gather = tf.gather_nd(w2, indices_tiled_2)

        # tensor_weights = broadcast_reshape(values_tiled, sparse_h_gather)

        sparse_h_weighed = tf.multiply(values_tiled_2, sparse_h_gather)
        sparse_h_weighed = tf.reduce_sum(sparse_h_weighed, 1)

        y = sparse_h_weighed + b2
        y = tf.reduce_sum(y)

        # sparse_h_gather = tf.reshape(sparse_h_gather,[n_partitions,2,1])
        # sparse_h_weighed = tf.multiply(sp_weights.values,sparse_h_gather)

        # TODO correct the operation to take both dense inputs and sparse inputs with batch 1,2,...

        # didn't know but calling global variable int must be done after adding the vars to the graph
        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)

            # feed_dict = {inputs: [[1., 0, -1]]}
            feed_dict = {inputs: [[1., 0, -1], [1., 0, -1]]}

            # print("neigh distances")
            # print(neigh_distances.eval(feed_dict))

            # print("weights:")
            # print(weights.eval(feed_dict))

            # print("weights:")
            # print(weights.eval(feed_dict))

            print("neight indices: ")
            print(neigh_indices.eval(feed_dict))

            # print("neigh weights:")
            # print(neigh_weights.eval(feed_dict))

            # print("neigh gather:")
            # print(w.eval(feed_dict))

            print("neigh gather * weights:")
            print(w2.eval(feed_dict))

            print("neigh gather * bias:")
            print(b2.eval(feed_dict))

            print("flat indices 2")
            print(flat_indices_2.eval(feed_dict))

            print("rows")
            print(rows.eval(feed_dict))

            print("rows_2")
            print(rows_2.eval(feed_dict))

            print("indices tiled 2")
            print(indices_tiled_2.eval(feed_dict))

            print("values 2")
            print(values_tiled_2.eval(feed_dict))

            print("new_dense_shape")
            print(dense_shape.eval(feed_dict))

            print("new indices")
            print(indices_tiled.eval(feed_dict))

            print("sparse h map_fn")
            print(sparse_h_map_fn.eval(feed_dict))

            print("sparse h gather")
            print(sparse_h_gather.eval(feed_dict))

            # print("spare h gather weights")
            # print(tensor_weights.eval(feed_dict))

            print("sparse h weights")
            print(sparse_h_weighed.eval(feed_dict))

            print("sparse y")
            print(y.eval(feed_dict))




            # print("som vars:\n", som_w.eval())
            # print("feature vars:\n", feature_w.eval())
            # print("dist: \n", som_dist.eval(feed_dict))
            # print(bmu.eval(feed_dict))

            # print("slices:\n", feature_w_slice.eval(feed_dict))
            # print("features:\n", features.eval(feed_dict))
            # hidden
            # print("result:\n", h.eval(feed_dict))

            # print(gaussian(0., 0.5).eval())

    def test_batch_som_indices(self):
        n_partitions = 4
        n_inputs = 3
        n_hidden = 1

        som_indices = tf.range(0, n_partitions)

        dense_x = tf.constant([[1., 0., -1.], [1., 0., -1.], [1., 1., 0.]])
        #dense_x = tf.constant([[1., 0., -1.]])
        sparse_x = transform.to_sparse(dense_x)

        som = tf.get_variable("som", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))
        all_w = tf.get_variable("w", [n_partitions, n_inputs, n_hidden], tf.float32,
                                tf.random_uniform_initializer(-1., 1.))

        init = tf.global_variables_initializer()
        self.ss.run(init)

        """ ************************************************************************************************************
        L1 Neighbourhood and BMU
        """

        distances = pairwise_cosine_distance(dense_x, som)
        bmu = tf.argmin(distances, axis=1)
        bmu_batch = tf.expand_dims(bmu, 1)
        bmu_rows = transform.batch_to_matrix_indices(bmu_batch)

        print("som: \n", som.eval())
        print("distances:\n", distances.eval())
        print("bmu: \n", bmu.eval())
        print("bmu batch: \n", bmu_rows.eval())

        print("winner distances:")
        winner_distances = tf.gather_nd(distances, bmu_rows)
        print(winner_distances.eval())

        som_l1 = l1_dist_wrap(bmu_batch, som_indices)
        print("l1 dist\n", som_l1.eval())

        """ ************************************************************************************************************
        Dynamic SOM Gaussian Neighbourhood
        """
        # sigma for gaussian is based on the distance between BMU (winner neuron) and X (data)
        elasticity = 1.2  # parameter for the dynamic SOM
        sigma = elasticity * winner_distances
        #sigma = tf.expand_dims(sigma,-1)

        #print("sigma: \n", sigma.eval())

        #gauss_neighbourhood = gaussian(som_l1, sigma=0.1)
        #print("gauss dist\n", gauss_neighbourhood.eval())

        gauss_neighbourhood = gaussian(som_l1, sigma)
        #print("dynamic gauss dist\n", gauss_neighbourhood.eval())
        gauss_threshold = 1e-6
        active_neighbourhood = tf.greater(gauss_neighbourhood, gauss_threshold)
        active_indices = tf.where(active_neighbourhood)

        #gauss_neighbourhood = transform.filter_nd(active_neighbourhood,gauss_neighbourhood)
        #or just clip to 0
        gauss_neighbourhood = tf.where(active_neighbourhood,gauss_neighbourhood,tf.zeros_like(active_neighbourhood,tf.float32))
        #print("dynamic gauss dist clipped\n", gauss_neighbourhood.eval())


        """ ************************************************************************************************************
        SOM Delta - how to change the som weights
        """
        learning_rate = 0.1
        delta = tf.expand_dims(dense_x,1) - som
        # since dense x is usually a batch, the delta should be the average
        delta = tf.reduce_mean(delta,axis=0)
        print("delta x - som_w \n", delta.eval())

        som_delta = learning_rate * distances #* gauss_neighbourhood * delta
        print("lr vs dist \n", som_delta.eval())
        som_delta *= gauss_neighbourhood
        print("delta_modulated by neihbourhood \n", som_delta.eval())
        som_delta = tf.expand_dims(som_delta, -1)
        som_delta *= delta
        # for a batch of changes take the mean of the deltas
        print("delta_modulated by average code-diff \n", som_delta.eval())
        som_delta = tf.reduce_mean(som_delta,0)
        print("delta_modulated by average code-diff \n", som_delta.eval())


        #gauss_neighbourhood = tf.boolean_mask(gauss_neighbourhood, active_indices)



        """ ************************************************************************************************************
                    
        """



        # TODO review index selection and weights to create a batch of indices ?
        # this should result in a sparse tensor since we might have different indices active for different layers
        #som_active_indices = tf.where(active_neighbourhood)
        #print("active som indices: \n", som_active_indices.eval())
        #gauss_neighbourhood = tf.boolean_mask(gauss_neighbourhood, active_neighbourhood)
        #print("som index weight: \n", gauss_neighbourhood.eval())

        # still need an expression to weight the different partitions which depends on
        # both the gaussian distances and the distances to the data of each partition neuron


if __name__ == '__main__':
    unittest.main()
