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


class Layer:
    def __init__(self, n_units, shape=None, dtype=tf.float32, name="layer"):
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

        # has a y (tensor) member
        self.y = None


class Input(Layer):
    """ Input Layer

    Creates a placeholder to receive tensors with a given shape and data type.
    """

    def __init__(self, n_units, batch_size=None, dtype=tf.float32, name="input"):
        shape = [batch_size, n_units]
        super().__init__(n_units, shape, dtype, name)
        self.y = tf.placeholder(self.dtype, self.shape, self.name)


def input(n_units, batch_size=None, dtype=tf.float32, name="input"):
    shape = [batch_size, n_units]
    y = tf.placeholder(dtype, shape, name)
    return y


class IndexInput(Layer):
    """ IndexInput Layer
    creates an int32 placeholder with n_active int elements
    used with sparse layers to slice weight matrices
    """

    def __init__(self, n_units, n_active, batch_size=None, name="index_input"):
        shape = [batch_size, n_active]
        super().__init__(n_units, shape, tf.int32, name)

        self.n_active = n_active
        self.y = tf.placeholder(self.dtype, self.shape, self.name)

    def to_dense(self):
        """Converts the output tensor
        to a dense s with n_units
        """
        return tf.one_hot(self.y, self.n_units)


class SparseInput(Layer):
    def __init__(self, n_units, n_active, batch_size=None, name="sparse_input"):
        shape = [batch_size, n_active]
        super().__init__(n_units, shape, tf.float32, self.name)


class Dense(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 init=random_uniform,
                 weights=None,
                 activation=None,
                 bias=False,
                 dtype=tf.float32,
                 name="dense"):

        shape = [layer.n_units, n_units]
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
            y = tf.matmul(layer.tensor, self.weights)

            # y = xW + [b]
            if bias:
                layer.bias = tf.get_variable("b", initializer=tf.zeros([self.n_units]))
                y = tf.nn.bias_add(y, self.bias, name="a")

            # y = fn(xW + [b])
            if activation is not None:
                y = activation(y, name="fn")

            self.tensor = y


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
