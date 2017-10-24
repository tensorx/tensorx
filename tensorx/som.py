"""Operators to build and train SOM networks.

"""
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import IndexedSlices

from tensorx.training import Learner
from tensorx import transform
from tensorx.metrics import pairwise_sparse_cosine_distance, torus_l1_distance
from tensorx.math import gaussian


class DSOM_Learner(Learner):
    def __init__(self, som_shape, learning_rate=0.1, elasticity=1.0, distance_metric=pairwise_sparse_cosine_distance,
                 neighbourhood_threshold=1e-6):
        """ The variable might be a flat version of the som_shape
        usually with som_shape[0]*som_shape[1] neurons but the actual
        shape of the som network needs to be known a prior to compute the neighbourhood
        function.

        Args:
            som_shape: a list with the shape of the som variable to which this learner will apply updates
            neighbourhood_threshold: if a unit falls bellow this neighbourhood threshold, its not updated
        """
        super().__init__()
        self.som_shape = som_shape
        self.distance_metric = distance_metric
        self.learning_rate = learning_rate
        self.elasticity = elasticity
        self.neighbourhood_threshold = neighbourhood_threshold

    def _compute_delta(self, data, var):
        som_shape = self.som_shape

        # pre-compute the indices for the som grid
        som_indices = transform.indices(som_shape)
        distances = self.distance_metric(data, var)

        # compute best-matching unit
        bmu = math_ops.argmin(distances, axis=1)
        bmu = array_ops.expand_dims(bmu, 1)

        bmu_indices = array_ops.gather_nd(som_indices, bmu)

        #print("bmu\n", bmu.eval())
        #print("bmu_indices \n", bmu_indices.eval())

        # compute l1 distances between units
        som_l1 = torus_l1_distance(bmu_indices, som_shape)

        bmu_rows = transform.batch_to_matrix_indices(bmu)
        bmu_distances = array_ops.gather_nd(distances, bmu_rows)
        sigma = self.elasticity * bmu_distances
        neighbourhood = gaussian(som_l1, sigma)
        neighbourhood_filter = math_ops.greater(neighbourhood, self.neighbourhood_threshold)
        active_indices = array_ops.where(neighbourhood_filter)

        # sparse tensor with neighbourhood activations
        # neighbourhood = transform.filter_nd(neighbourhood_filter,neighbourhood)
        neighbourhood = array_ops.where(neighbourhood_filter, neighbourhood,
                                        array_ops.zeros_like(neighbourhood_filter, dtypes.float32))

        # compute delta based on neighbourhood and distance between data and codebook of som
        som_delta = self.learning_rate * distances * neighbourhood
        som_delta = array_ops.expand_dims(som_delta, -1)

        delta = array_ops.expand_dims(data, 1) - var
        delta = math_ops.reduce_mean(delta, axis=0)
        som_delta *= delta
        som_delta = math_ops.reduce_mean(som_delta, 0)

        sp_delta = transform.to_sparse(som_delta)
        sp_delta = IndexedSlices(sp_delta.indices, sp_delta.values, sp_delta.dense_shape)

        return sp_delta

    def compute_delta(self, data, var_list):
        deltas_and_vars = []
        for var in var_list:
            delta = self._compute_delta(data, var)
            deltas_and_vars.append((delta, var))

        return deltas_and_vars
