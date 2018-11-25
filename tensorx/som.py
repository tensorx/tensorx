"""Operators to build and train SOM networks.

"""
from tensorflow.python.ops import array_ops, math_ops, sparse_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.train import Learner
from tensorx import transform
from tensorx import metrics
from tensorx.math import gaussian
from tensorflow import sparse


class DSOMLearner(Learner):
    def __init__(self, var_list, som_shape, learning_rate=0.1, elasticity=1.0, neighbourhood_threshold=1e-6,
                 metric=metrics.batch_sparse_cosine_distance):
        """ The variable might be a flat version of the som_shape
        usually with som_shape[0]*som_shape[1] neurons but the actual
        shape of the som network needs to be known a prior to compute the neighbourhood
        function.

        Warning:
            the distance metric has to be compatible with the type of tensor for the given data. Either pass
            a metric that can handle both `Tensor` and `SparseTensor` instances, or make sure the metric, corresponds
            to the data that is fed to the learner.

        Args:
            som_shape: a list with the shape of the som variable to which this learner will apply updates
            neighbourhood_threshold: if a unit falls bellow this neighbourhood threshold, its not updated
        """
        super().__init__(var_list)
        self.som_shape = som_shape
        self.learning_rate = learning_rate
        self.elasticity = elasticity
        self.neighbourhood_threshold = neighbourhood_threshold
        self.metric = metric

    def _compute_delta(self, data, var):
        codebook = var
        som_shape = self.som_shape

        # distances
        som_indices = transform.grid(som_shape)
        distances = self.metric(data, codebook)

        max_dist = math_ops.reduce_max(distances, axis=1, keepdims=True)
        norm_dist = distances / max_dist

        # l1 distance to winner neuron
        bmu = math_ops.argmin(distances, axis=1)
        bmu = array_ops.expand_dims(bmu, 1)
        bmu_rows = transform.to_matrix_indices(bmu)
        winner_indices = array_ops.gather_nd(som_indices, bmu)
        som_l1 = metrics.torus_l1_distance(winner_indices, som_shape)

        # GAUSS NEIGHBOURHOOD (sigma = elasticity * dist(winner))
        winner_distances = array_ops.gather_nd(distances, bmu_rows)
        sigma = self.elasticity * winner_distances
        gauss_neighbourhood = gaussian(som_l1, sigma)

        clip_neighbourhood = math_ops.greater(gauss_neighbourhood, self.neighbourhood_threshold)
        gauss_neighbourhood = array_ops.where(clip_neighbourhood, gauss_neighbourhood,
                                              array_ops.zeros_like(gauss_neighbourhood, dtypes.float32))

        lr = self.learning_rate * norm_dist * gauss_neighbourhood

        # delta x - som
        if isinstance(data, SparseTensor):
            data = sparse.to_dense(data)
        delta = array_ops.expand_dims(data, 1) - codebook
        delta = array_ops.expand_dims(lr, -1) * delta

        # take the mean of each delta for each sample in a batch
        delta = math_ops.reduce_mean(delta, 0)

        return delta

    def compute_delta(self, data):
        deltas_and_vars = []
        for var in self.var_list:
            delta = self._compute_delta(data, var)
            deltas_and_vars.append((delta, var))

        return deltas_and_vars
