""" Neural Network Layers.

All layers contain a certain number of units, its shape, name and a tensor member
which gives us a handle for a TensorFlow tensor that can be evaluated.

Types of layers:
    input: wrap around TensorFlow placeholders.

    dense:  a layer encapsulating a dense matrix of weights,
            possibly including biases and an activation function.

    sparse: a dense matrix of weights accessed through a list of indexes,
            (e.g. by being connected to an IndexInput layer)

    merge: utility to merge other layers

    bias: adds a bias to a given layer with its own scope for the bias variable

    activation: adds an activation function to a given layer with its own scope
"""

import tensorflow as tf
from tensorflow.python.ops import random_ops
import numbers

from tensorx.init import random_uniform


class Layer:
    def __init__(self, n_units, shape=None, dense_shape=None, dtype=tf.float32, name="layer"):
        """
        Args:
            n_units: dimension of input vector (dimension of columns in case batch_size != None
            shape: [batch size, input dimension]
            dtype: expected input TensorFlow data type
            name: layer name (used to nam the placeholder)
        """
        self.n_units = n_units
        self.name = name
        self.dtype = dtype

        if shape is None:
            self.shape = [None, n_units]
        else:
            self.shape = shape

        if dense_shape is None:
            self.dense_sape = self.shape
        else:
            if dense_shape[1] < n_units:
                raise Exception("Shape mismatch: dense_shape[1] < n_units")
            elif dense_shape[0] != self.shape[0]:
                raise Exception("Shape mismatch: dense_shape[0] != self.shape[0]")
            else:
                self.dense_sape = dense_shape

        # has a y (tensor) attribute
        self.y = None


class Input(Layer):
    """ Input Layer

    Creates a placeholder to receive tensors with a given shape and data type.
    """

    def __init__(self, n_units, n_active=None, batch_size=None, dense_shape=None, dtype=tf.float32, name="input"):
        if n_active is not None and n_active >= n_units:
            dense_shape = [batch_size, n_units]
            shape = [batch_size, n_active]

        super().__init__(n_units, shape, dense_shape, batch_size, dtype, name)
        self.y = tf.placeholder(self.dtype, self.shape, self.name)


class SparseInput(Layer):
    """ Sparse Input Layer
    creates an int32 placeholder with n_active int elements and
    a float32 placeholder for values corresponding to each index

    USE CASE:
        used with sparse layers to slice weight matrices
        alternatively each slice can be weighted by the given values

    Args:
        values - if true, creates a sparse placeholder with (indices, values)

    Placeholders:
        indices = instead of [[0],[2,5],...] -> SparseTensorValue([[0,0],[1,2],[1,5]],[0,2,5])
        values = [0.2,0.0,2.0] -> SparseTensorValue([[0,0],[1,2],[1,5]],[0.2,0.0,2.0])

    Note:
        See the following utils:

        tensorx.utils.data.index_list_to_sparse
        tensorx.utils.data.value_list_to_sparse
    """

    def __init__(self, n_units, n_active, values=False, batch_size=None, dtype=tf.float32, name="index_input"):
        shape = [batch_size, n_active]
        dense_shape = [batch_size, n_units]
        super().__init__(n_units, shape, dense_shape, batch_size, dtype, name)

        self.n_active = n_active
        self.values = values

        self.indices = tf.sparse_placeholder(tf.int64, self.shape, name)

        if values:
            self.values = tf.sparse_placeholder(dtype, self.shape, name=name + "_values")
        else:
            self.values = None

        self.y = tf.SparseTensor(self.indices, self.values, self.dense_sape)


class Dense(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 init=random_uniform,
                 weights=None,
                 act_fn=None,
                 bias=False,
                 dtype=tf.float32,
                 name="dense"):

        shape = [layer.dense_shape[0], n_units]
        super().__init__(n_units, shape, dtype, name)

        # if weights are passed, check that their shape matches the layer shape
        if weights is not None:
            (_, s) = weights.get_shape()
            if s != n_units:
                raise ValueError("shape mismatch: layer expects (,{}), weights have (,{})".format(n_units, s))

        with tf.variable_scope(name):
            # init weights
            if weights is not None:
                self.weights = weights
            else:
                self.weights = tf.get_variable("w", initializer=init(self.shape))

            # y = xW
            if hasattr(layer, "sp_indices"):
                indices = getattr(layer, "sp_indices")
                values = getattr(layer, "sp_values", default=None)

                lookup_sum = tf.nn.embedding_lookup_sparse(params=self.weights,
                                                           sp_ids=indices,
                                                           sp_weights=values,
                                                           combiner="sum",
                                                           name=self.name + "_embeddings")
                self.y = lookup_sum
            else:
                if layer.shape == layer.dense_shape:
                    self.y = tf.matmul(layer.y, self.weights)
                else:
                    lookup = tf.nn.embedding_lookup(params=self.weights,
                                                    ids=layer.y,
                                                    name=self.name + "_embeddings")
                    self.y = tf.reduce_sum(lookup, axis=1)

            # y = xW + [b]
            if bias:
                self.bias = tf.get_variable("b", initializer=tf.zeros([self.n_units]))
                self.y = tf.nn.bias_add(self.y, self.bias, name="a")

            self.logits = self.y
            # y = fn(xW + [b])
            if act_fn is not None:
                self.act_fn = act_fn
                self.y = act_fn(self.y, name="act_fn")


class ToSparse(Layer):
    """ Transforms the previous layer into a sparse representation

    meaning the current layer provides:
        sp_indices
        sp_values
    """

    def __init__(self, layer):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_sparse")

        with tf.name_scope(self.name):
            indices = tf.where(tf.not_equal(layer.y, 0))
            dense_shape = tf.shape(layer.y, out_type=tf.int64)

            # Sparse Tensor for sp_indices
            flat_layer = tf.reshape(layer.y, [-1])
            values = tf.mod(tf.squeeze(tf.where(tf.not_equal(flat_layer, 0))), layer.n_units)

            self.sp_indices = tf.SparseTensor(indices, values, dense_shape)

            # Sparse Tensor for values
            values = tf.gather_nd(layer.y, indices)
            self.sp_values = tf.SparseTensor(indices, values, dense_shape)


class GaussianNoise:
    def __call__(self, layer,std):
        def gaussian_noise_layer(input_layer, std):
            noise = tf.random_normal(shape=input_layer.y, mean=0.0, stddev=std, dtype=tf.float32)
            return input_layer + noise




class Noise(Layer):
    def __init__(self, layer, noise_type="gaussian", amount=0.5, seed=None, **kargs):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_noise")
        with tf.name_scope(self.name):
            if isinstance(amount, numbers.Real) and not 0 < amount <= 1:
                raise ValueError("amount must be a scalar tensor or a float in the "
                                 "range (0, 1], got %g" % amount)

            # do nothing if amount of noise is 0
            if amount == 0:
                self.y = layer.y
            else:
                random_ops.random_normal(tf.shape(layer.y))

                noise = tf.random_normal(shape=tf.shape(layer.y), mean=0.0, stddev=std, dtype=tf.float32)
                return input_layer + noise






class Bias(Layer):
    """ Bias Layer

    A simple way to add a bias to a given layer, the dimensions of this variable
    are determined by the given layer and it is initialised with zeros
    """

    def __init__(self, layer, name="bias"):
        bias_name = layer.dtype, "{}_{}".format(layer.name, name)
        super().__init__(layer.n_units, layer.shape, bias_name)

        with tf.variable_scope(self.name):
            self.bias = tf.get_variable("b", initializer=tf.zeros([self.n_units]))
            self.tensor = tf.nn.bias_add(layer.tensor, self.bias, name="output")


class Merge(Layer):
    """Merge Layer

    Merges a list layers by combining their tensors with a merging function.
    Allows for the output of each layer to be weighted.

    This is just a container that for convenience takes the output of each given layer (which is generaly a tensor),
    and applies a merging function.
    """

    def __init__(self,
                 layers,
                 weights=None,
                 merge_fn=tf.add_n,
                 name="merge"):
        """
        Args:
            layers: a list of layers with the same number of units to be merged
            weights: a list of weights
            merge_fn: must operate on a list of tensors
            name: name for layer which creates a named-scope

        Requires:
            len(layers) == len(weights)
            layer[0].n_units == layer[1].n_units ...
            layer[0].dtype = layer[1].dtype ...
        """
        if len(layers) < 2:
            raise Exception("Expecting a list of layers with len >= 2")

        if weights is not None and len(weights) != len(layers):
            raise Exception("len(weights) must be equals to len(layers)")

        super().__init__(layers[0].n_units, layers[0].shape, layers[0].dense_shape, layers[0].dtype, name)

        with tf.name_scope(name):
            if weights is not None:
                for i in range(len(layers)):
                    layers[i] = tf.scalar_mul(weights[i], layers[i].output)

            self.y = merge_fn(layers)


