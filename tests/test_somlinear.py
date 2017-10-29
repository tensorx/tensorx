import unittest
import tensorflow as tf
from tensorx.metrics import pairwise_cosine_distance, torus_l1_distance, pairwise_sparse_cosine_distance

import numpy as np

from tensorx import transform
from tensorx.math import gaussian, sparse_mul
from tensorx.som import DSOM_Learner


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
        neigh_distances = tf.to_float(torus_l1_distance(bmu, [n_partitions]))
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

    def test_som_2d(self):
        som_shape = [2, 2]
        n_partitions = som_shape[0] * som_shape[1]
        n_inputs = 2
        n_hidden = 1

        inputs = tf.constant([[1., 0.], [1., 1.]])
        sp_inputs = transform.to_sparse(inputs)

        # the som variable can be flatten since we have the ordered distances
        som = tf.get_variable("som", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))
        w = tf.get_variable("w", [n_partitions, n_inputs, n_hidden], tf.float32, tf.random_uniform_initializer(-1., 1.))

        init = tf.global_variables_initializer()
        self.ss.run(init)

        # **************************************************************************************************************
        # SOM Neighbourhood - GAUSSIAN(L1(GRID))
        # **************************************************************************************************************
        som_indices = transform.indices(som_shape)
        distances = pairwise_cosine_distance(inputs, som)
        print(distances.eval())
        bmu = tf.argmin(distances, axis=1)
        bmu = tf.expand_dims(bmu, 1)

        print(bmu.eval())

        bmu_rows = transform.batch_to_matrix_indices(bmu)
        winner_distances = tf.gather_nd(distances, bmu_rows)

        winner_indices = tf.gather_nd(som_indices, bmu)
        print("winner indices \n", winner_indices.eval())
        som_l1 = torus_l1_distance(winner_indices, som_shape)
        print(som_l1.eval())

        learning_rate = 0.1
        elasticity = 1.2
        neighbourhood_threshold = 1e-6
        dsom_learner = DSOM_Learner(som_shape=som_shape,
                                    learning_rate=learning_rate,
                                    elasticity=elasticity,
                                    distance_metric=pairwise_cosine_distance,
                                    neighbourhood_threshold=neighbourhood_threshold)

        deltas = dsom_learner.compute_delta(inputs, [som])

    def test_batch_som_indices(self):
        n_partitions = 4
        n_inputs = 2
        n_hidden = 1
        som_shape = [n_partitions]

        som_indices = tf.range(0, n_partitions)

        inputs = tf.constant([[1., 0.], [1., 1.]])
        # inputs = tf.constant([[1., 0., -1.]])
        sp_inputs = transform.to_sparse(inputs)

        som = tf.get_variable("som", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))
        all_w = tf.get_variable("w", [n_partitions, n_inputs, n_hidden], tf.float32,
                                tf.random_uniform_initializer(-1., 1.))

        init = tf.global_variables_initializer()
        self.ss.run(init)

        """ ************************************************************************************************************
        L1 Neighbourhood and BMU
        """

        distances = pairwise_cosine_distance(inputs, som)
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

        print("bmu_batch\n", bmu_batch)
        som_l1 = torus_l1_distance(bmu_batch, som_shape)
        print("l1 dist\n", som_l1.eval())

        """ ************************************************************************************************************
        Dynamic SOM Gaussian Neighbourhood
        """

        # sigma for gaussian is based on the distance between BMU (winner neuron) and X (data)
        elasticity = 1.2  # parameter for the dynamic SOM
        sigma = elasticity * winner_distances
        # sigma = tf.expand_dims(sigma,-1)

        # print("sigma: \n", sigma.eval())

        # gauss_neighbourhood = gaussian(som_l1, sigma=0.1)
        # print("gauss dist\n", gauss_neighbourhood.eval())

        gauss_neighbourhood = gaussian(som_l1, sigma)
        # print("dynamic gauss dist\n", gauss_neighbourhood.eval())
        gauss_threshold = 1e-6
        active_neighbourhood = tf.greater(gauss_neighbourhood, gauss_threshold)
        active_indices = tf.where(active_neighbourhood)

        sp_gauss_neighbourhood = transform.filter_nd(active_neighbourhood, gauss_neighbourhood)
        # or just clip to 0
        gauss_neighbourhood = tf.where(active_neighbourhood, gauss_neighbourhood,
                                       tf.zeros_like(active_neighbourhood, tf.float32))

        self.assertTrue(
            np.array_equal(tf.sparse_tensor_to_dense(sp_gauss_neighbourhood).eval(), gauss_neighbourhood.eval()))
        # print("dynamic gauss dist clipped\n", gauss_neighbourhood.eval())



        learning_rate = 0.1
        delta = tf.expand_dims(inputs, 1) - som
        # since dense x is usually a batch, the delta should be the average
        delta = tf.reduce_mean(delta, axis=0)
        # print("delta x - som_w \n", delta.eval())

        som_delta = learning_rate * distances  # * gauss_neighbourhood * delta
        # print("lr vs dist \n", som_delta.eval())
        som_delta *= gauss_neighbourhood

        sp_som_delta = sparse_mul(sp_gauss_neighbourhood, som_delta)
        _, sp_slices = tf.unstack(sp_som_delta.indices, num=2, axis=-1)
        print("sp_slices\n", sp_slices.eval())

        print("sp_delta\n", sp_som_delta.eval())

        gathered = tf.nn.embedding_lookup(delta, sp_slices)
        print("gathered \n", gathered.eval())

        gathered_mul = tf.expand_dims(sp_som_delta.values, 1) * gathered

        # avg = tf.segment_mean(gathered,sp_slices)
        print("weighted gathered \n", gathered_mul.eval())
        # TODO THIS DOESN'T WORK, I NEED TO SEE HOW INDEXED SLICES WORK
        num_indices, _ = tf.unique(sp_slices)  # tf.shape(tf.unique(sp_slices))[0]
        num_indices = tf.shape(num_indices)[0]
        # print("uniqyue \n", num_indices.eval())
        ones_indices = tf.ones_like(gathered_mul)
        count = tf.unsorted_segment_sum(ones_indices, sp_slices, num_indices)
        # gathered_sum = tf.unsorted_segment_sum(gathered_mul, sp_slices,num_indices)
        # gathered_mean = tf.divide(gathered_sum,count)
        # print("sum gathered \n", gathered_sum.eval())


        # print("delta_modulated by neihbourhood \n", som_delta.eval())
        # print("delta_modulated by sp neihbourhood \n", sp_som_delta.eval())

        som_delta = tf.expand_dims(som_delta, -1)
        som_delta *= delta
        som_delta = tf.reduce_mean(som_delta, 0)

        print("final delta \n", som_delta.eval())

        op = tf.assign_sub(som, som_delta)

        print("final som \n", op.eval())

        learning_rate = 0.1
        elasticity = 1.2
        neighbourhood_threshold = 1e-6
        dsom_learner = DSOM_Learner(som_shape=som_shape,
                                    learning_rate=learning_rate,
                                    elasticity=elasticity,
                                    distance_metric=pairwise_cosine_distance,
                                    neighbourhood_threshold=neighbourhood_threshold)

        deltas = dsom_learner.compute_delta(inputs, [som])

        # print("final updates learner\n", dsom_learner.apply_delta(deltas).eval())


        # print("learner deltas\n", deltas[0].eval())



        """ ************************************************************************************************************
                    
        """



        # TODO review index selection and weights to create a batch of indices ?
        # this should result in a sparse tensor since we might have different indices active for different layers
        # som_active_indices = tf.where(active_neighbourhood)
        # print("active som indices: \n", som_active_indices.eval())
        # gauss_neighbourhood = tf.boolean_mask(gauss_neighbourhood, active_neighbourhood)
        # print("som index weight: \n", gauss_neighbourhood.eval())

        # still need an expression to weight the different partitions which depends on
        # both the gaussian distances and the distances to the data of each partition neuron

    def test_delta_computing(self):
        n_partitions = 4
        n_inputs = 3
        n_hidden = 1
        som_shape = [n_partitions]
        learning_rate = 0.1
        elasticity = 1.2
        neighbourhood_threshold = 1e-6

        inputs = tf.constant([[1., 0., 0.], [0., 1., 0.]])
        som = tf.get_variable("som", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))

        init = tf.global_variables_initializer()
        self.ss.run(init)

        # DISTANCES
        som_indices = transform.indices(som_shape)
        if isinstance(inputs, tf.Tensor):
            distances = pairwise_cosine_distance(inputs, som)
        elif isinstance(inputs, tf.SparseTensor):
            distances = pairwise_sparse_cosine_distance(inputs, som)
        else:
            raise TypeError("invalid inputs")
        # print("distances \n ",distances.eval())
        max_dist = tf.reduce_max(distances, axis=1, keep_dims=True)
        # print(max_dist.eval())
        norm_dist = distances / max_dist
        # print("normalised distances ",  norm_dist.eval())
        print("dist shape: ", tf.shape(norm_dist).eval())

        # WINNER l1 DISTANCE
        bmu = tf.argmin(distances, axis=1)
        bmu = tf.expand_dims(bmu, 1)
        bmu_rows = transform.batch_to_matrix_indices(bmu)
        winner_indices = tf.gather_nd(som_indices, bmu)

        som_l1 = torus_l1_distance(winner_indices, som_shape)
        print("l1 shape: ", tf.shape(som_l1).eval())
        # print(som_l1.eval())

        # GAUSS NEIGHBOURHOOD (sigma = elasticity * dist(winner))
        winner_distances = tf.gather_nd(distances, bmu_rows)
        sigma = elasticity * winner_distances
        gauss_neighbourhood = gaussian(som_l1, sigma)
        print("gauss shape: ", tf.shape(gauss_neighbourhood).eval())
        clip_neighbourhood = tf.greater(gauss_neighbourhood, neighbourhood_threshold)
        gauss_neighbourhood = tf.where(clip_neighbourhood, gauss_neighbourhood,
                                       tf.zeros_like(gauss_neighbourhood, tf.float32))

        lr = learning_rate * norm_dist * gauss_neighbourhood
        print("lr * norm_dist * gauss_neighbourhood ", tf.shape(lr).eval())
        print(lr.eval())

        # delta x - som
        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse_tensor_to_dense(inputs)
        delta = tf.expand_dims(inputs, 1) - som
        print("shape delta x - som ", tf.shape(delta).eval())
        # print("in \n", inputs.eval())
        # print("som \n", som.eval())
        print("delta \n", delta.eval())

        delta = tf.expand_dims(lr, -1) * delta
        print("shape lr * delta ", tf.shape(delta).eval())
        print(delta.eval())

        print(tf.reduce_mean(delta,0).eval())









if __name__ == '__main__':
    unittest.main()
