import tensorflow as tf
from tensorx import test_utils
from tensorx.metrics import batch_cosine_distance, torus_l1_distance, batch_sparse_cosine_distance

import numpy as np

from tensorx import transform
from tensorx.math import gaussian, sparse_multiply
from tensorx.som import DSOMLearner
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


class MyTestCase(test_utils.TestCase):

    def test_sate_gauss_neighbourhood(self):
        result1 = gaussian(0., 0.5)
        result2 = gaussian([-2., -1., 0., 1., 2.], 2.)

        with self.cached_session(use_gpu=True):
            self.assertEqual(result1, 1.)
            self.assertArrayEqual(result2[0:2], result2[3:5][::-1])


    def test_som_2d(self):
        tf.reset_default_graph()
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

        # **************************************************************************************************************
        # SOM Neighbourhood - GAUSSIAN(L1(GRID))
        # **************************************************************************************************************
        som_indices = transform.grid(som_shape)
        distances = batch_cosine_distance(inputs, som)

        bmu = tf.argmin(distances, axis=1)
        bmu = tf.expand_dims(bmu, 1)

        bmu_rows = transform.to_matrix_indices(bmu)
        winner_distances = tf.gather_nd(distances, bmu_rows)

        winner_indices = tf.gather_nd(som_indices, bmu)

        som_l1 = torus_l1_distance(winner_indices, som_shape)

        learning_rate = 0.1
        elasticity = 1.2
        neighbourhood_threshold = 1e-6
        dsom_learner = DSOMLearner(var_list=[],
                                   som_shape=som_shape,
                                   learning_rate=learning_rate,
                                   elasticity=elasticity,
                                   metric=batch_cosine_distance,
                                   neighbourhood_threshold=neighbourhood_threshold)

        deltas = dsom_learner.compute_delta(inputs)

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.eval(deltas)

    def test_batch_som_indices(self):
        tf.reset_default_graph()
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

        """ 
        L1 Neighbourhood and BMU
        """

        distances = batch_cosine_distance(inputs, som)
        bmu = tf.argmin(distances, axis=1)
        bmu_batch = tf.expand_dims(bmu, 1)
        bmu_rows = transform.to_matrix_indices(bmu_batch)

        # print("som: \n", som.eval())
        # print("distances:\n", distances.eval())
        # print("bmu: \n", bmu.eval())
        # print("bmu batch: \n", bmu_rows.eval())

        # print("winner distances:")
        winner_distances = tf.gather_nd(distances, bmu_rows)
        # print(winner_distances.eval())

        # print("bmu_batch\n", bmu_batch)
        som_l1 = torus_l1_distance(bmu_batch, som_shape)
        # print("l1 dist\n", som_l1.eval())

        """ 
        Dynamic SOM Gaussian Neighbourhood
        """

        # sigma for gaussian is based on the distance between BMU (winner neuron) and X (data)
        elasticity = 1.2  # parameter for the dynamic SOM
        sigma = elasticity * winner_distances

        gauss_neighbourhood = gaussian(som_l1, sigma)
        # print("dynamic gauss dist\n", gauss_neighbourhood.eval())
        gauss_threshold = 1e-6
        active_neighbourhood = tf.greater(gauss_neighbourhood, gauss_threshold)
        active_indices = tf.where(active_neighbourhood)

        sp_gauss_neighbourhood = transform.filter_nd(active_neighbourhood, gauss_neighbourhood)
        # or just clip to 0
        gauss_neighbourhood = tf.where(active_neighbourhood, gauss_neighbourhood,
                                       tf.zeros_like(active_neighbourhood, tf.float32))

        with self.cached_session(use_gpu=True):
            self.eval(init)

            dense_gauss_neighbourhood = tf.sparse.to_dense(sp_gauss_neighbourhood)
            self.assertArrayEqual(dense_gauss_neighbourhood, gauss_neighbourhood)

        # print("dynamic gauss dist clipped\n", gauss_neighbourhood.eval())

        learning_rate = 0.1
        delta = tf.expand_dims(inputs, 1) - som
        # since dense x is usually a batch, the delta should be the average
        delta = tf.reduce_mean(delta, axis=0)
        # print("delta x - som_w \n", delta.eval())

        som_delta = learning_rate * distances  # * gauss_neighbourhood * delta
        # print("lr vs dist \n", som_delta.eval())
        som_delta *= gauss_neighbourhood

        sp_som_delta = sparse_multiply(sp_gauss_neighbourhood, som_delta)
        _, sp_slices = tf.unstack(sp_som_delta.indices, num=2, axis=-1)
        # print("sp_slices\n", sp_slices.eval())

        # print("sp_delta\n", sp_som_delta.eval())

        gathered = tf.nn.embedding_lookup(delta, sp_slices)
        # print("gathered \n", gathered.eval())

        gathered_mul = tf.expand_dims(sp_som_delta.values, 1) * gathered

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

        # print("final delta \n", som_delta.eval())

        op = tf.assign_sub(som, som_delta)

        # print("final som \n", op.eval())

        learning_rate = 0.1
        elasticity = 1.2
        neighbourhood_threshold = 1e-6
        dsom_learner = DSOMLearner(var_list=[som],
                                   som_shape=som_shape,
                                   learning_rate=learning_rate,
                                   elasticity=elasticity,
                                   metric=batch_cosine_distance,
                                   neighbourhood_threshold=neighbourhood_threshold)

        deltas = dsom_learner.compute_delta(inputs)

        with self.cached_session(use_gpu=True):
            self.eval(deltas[0][0])

    def test_delta_computing(self):
        n_partitions = 4
        n_inputs = 3
        som_shape = [n_partitions]
        learning_rate = 0.1
        elasticity = 1.2
        neighbourhood_threshold = 1e-6

        inputs = tf.constant([[1., 0., 0.], [0., 1., 0.]])
        som = tf.get_variable("som", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))

        init = tf.global_variables_initializer()
        # DISTANCES
        som_indices = transform.grid(som_shape)
        if isinstance(inputs, tf.Tensor):
            distances = batch_cosine_distance(inputs, som)
        elif isinstance(inputs, tf.SparseTensor):
            distances = batch_sparse_cosine_distance(inputs, som)
        else:
            raise TypeError("invalid inputs")
        # print("distances \n ",distances.eval())
        max_dist = tf.reduce_max(distances, axis=1, keepdims=True)
        # print(max_dist.eval())
        norm_dist = distances / max_dist
        # print("normalised distances ",  norm_dist.eval())
        # print("dist shape: ", tf.shape(norm_dist).eval())

        # WINNER l1 DISTANCE
        bmu = tf.argmin(distances, axis=1)
        bmu = tf.expand_dims(bmu, 1)
        bmu_rows = transform.to_matrix_indices(bmu)
        winner_indices = tf.gather_nd(som_indices, bmu)

        som_l1 = torus_l1_distance(winner_indices, som_shape)
        # print("l1 shape: ", tf.shape(som_l1).eval())
        # print(som_l1.eval())

        # GAUSS NEIGHBOURHOOD (sigma = elasticity * dist(winner))
        winner_distances = tf.gather_nd(distances, bmu_rows)
        sigma = elasticity * winner_distances
        gauss_neighbourhood = gaussian(som_l1, sigma)
        # print("gauss shape: ", tf.shape(gauss_neighbourhood).eval())
        clip_neighbourhood = tf.greater(gauss_neighbourhood, neighbourhood_threshold)
        gauss_neighbourhood = tf.where(clip_neighbourhood, gauss_neighbourhood,
                                       tf.zeros_like(gauss_neighbourhood, tf.float32))

        lr = learning_rate * norm_dist * gauss_neighbourhood
        # print("lr * norm_dist * gauss_neighbourhood ", tf.shape(lr).eval())
        # print(lr.eval())

        # delta x - som
        if isinstance(inputs, tf.SparseTensor):
            inputs = tf.sparse.to_dense(inputs)
        delta = tf.expand_dims(inputs, 1) - som
        # print("shape delta x - som ", tf.shape(delta).eval())
        # print("in \n", inputs.eval())
        # print("som \n", som.eval())
        # print("delta \n", delta.eval())

        delta = tf.expand_dims(lr, -1) * delta
        # print("shape lr * delta ", tf.shape(delta).eval())
        # print(delta.eval())

        delta = tf.reduce_mean(delta, 0)

        dsom_learner = DSOMLearner(var_list=[som],
                                   som_shape=som_shape,
                                   learning_rate=learning_rate,
                                   elasticity=elasticity,
                                   metric=batch_cosine_distance,
                                   neighbourhood_threshold=neighbourhood_threshold)

        deltas = dsom_learner.compute_delta(inputs)

        delta_learner, learning_vars = deltas[0]

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(learning_vars, som)
            self.assertArrayEqual(delta, delta_learner)

        dsom_learner.adapt_to([inputs])

        # som2 = tf.identity(som)
        # print("som 2", som2.eval())


if __name__ == '__main__':
    test_utils.main()
