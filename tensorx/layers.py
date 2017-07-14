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

from tensorx.init import random_uniform

""" 
************************************************************************************************************************
LAYERS
************************************************************************************************************************
"""


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

        # has a y (tensor) member
        self.y = None

#TODO don't know if this works like this, check
class SparseLayer(Layer):
    def __init__(self, **kargs):
        super().__init__(kargs)
        self.indices = None
        self.values = None


class Input(Layer):
    """ Input Layer

    Creates a placeholder to receive tensors with a given shape and data type.
    """

    def __init__(self, n_units, n_active=None, batch_size=None, dense_shape=None, dtype=tf.float32, name="input"):
        if n_active is not None and n_active >= n_units:
            dense_sape = [batch_size, n_units]
            shape = [batch_size, n_active]
        else:
            shape = [batch_size, n_units]
            dense_sape = shape

        super().__init__(n_units, shape, dense_sape, dtype, name)
        self.y = tf.placeholder(self.dtype, self.shape, self.name)


class SparseInput(SparseLayer):
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
        super().__init__(n_units, shape, dense_shape, dtype, name)

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
                 input_layer,
                 n_units,
                 init=random_uniform,
                 weights=None,
                 activation=None,
                 bias=False,
                 dtype=tf.float32,
                 name="dense"):

        shape = [input_layer.dense_shape[0], n_units]
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

            x = input_layer.y
            # y = xW
            if isinstance(input_layer, SparseLayer):
                lookup_sum = tf.nn.embedding_lookup_sparse(params=self.weights,
                                                       sp_ids=x.indices,
                                                       sp_weights=x.values,
                                                       combiner="sum",
                                                       name=self.name + "_embeddings")
                y = lookup_sum
            else:
                if input_layer.shape == input_layer.dense_shape:
                    y = tf.matmul(x, self.weights)
                else:
                    lookup = tf.nn.embedding_lookup(params=self.weights,
                                                    ids=x,
                                                    name=self.name + "_embeddings")
                    y = tf.reduce_sum(lookup, axis=1)

            # y = xW + [b]
            if bias:
                input_layer.bias = tf.get_variable("b", initializer=tf.zeros([self.n_units]))
                y = tf.nn.bias_add(y, self.bias, name="a")

            # y = fn(xW + [b])
            if activation is not None:
                y = activation(y, name="fn")

            self.y = y


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

    Optional biases and activation function
    """

    def __init__(self,
                 layers,
                 weights=None,
                 merge_fn=tf.add_n,
                 name="merge"):
        """
        :param layers: a list of layers with the same number of units to be merged
        :param weights: a list of weights
        :param merge_fn: must operate on a list of tensors
        :param name: name for layer which creates a named-scope

        Requires:
            len(layers) == len(weights)
            layer[0].n_units == layer[1].n_units ...
            layer[0].dtype = layer[1].dtype ...
        """
        super().__init__(layers[0].n_units, layers[0].shape, layers[0].dtype, name)

        with tf.variable_scope(name):
            if weights is not None:
                for i in range(len(layers)):
                    layers[i] = tf.scalar_mul(weights[i], layers[i].output)

            self.tensor = merge_fn(layers)
