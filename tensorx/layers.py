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

import numbers
from tensorflow.python.framework import ops, tensor_util, tensor_shape
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops as control
from tensorflow.python.ops import check_ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vscope
from tensorflow.python.framework.ops import name_scope

from tensorflow.python.ops import random_ops
from tensorflow.python.ops.nn import embedding_lookup, embedding_lookup_sparse, bias_add, dropout
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.init import random_uniform
from tensorx.random import salt_pepper_noise, sparse_random_normal
import tensorx.transform as transform


class Layer:
    def __init__(self, n_units, shape=None, dtype=dtypes.float32, name="layer"):
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
            if shape[1] < n_units:
                raise ValueError("Shape mismatch: shape[1] < n_units")
            self.shape = shape

        # has an tensor (tensor) attribute
        self.tensor = None

    def is_sparse(self):
        return isinstance(self.tensor, SparseTensor)

    def _forward(self, prev_layer):
        """ Modifies the current layer with the relevant outputs
        from the given layer.

        Use Cases:
            when the current layer parameters define a transformation that does not
            affect the previous layer

            TODO I should probably issue a warning when this happens, so that users know what's going on
        Args:
            prev_layer: the layer to be forwarded in this layer

        Returns:

        """

        self.shape = prev_layer.shape
        self.tensor = prev_layer.tensor


class Input(Layer):
    """ Input Layer

    Creates a placeholder to receive tensors with a given shape and data type.
    """

    def __init__(self, n_units, n_active=None, batch_size=None, dtype=dtypes.float32, name="input"):
        """
        if n_active is not None:
            when connected to a Linear layer, this is interpreted
            as a binary sparse input layer and the linear layer is constructed using the
            Embedding Lookup operator.

            expects: int64 as inputs

        Note on sparse inputs:
            if you want to feed a batch of sparse binary features with weights, use SparseInput instead

        Args:
            n_units: number of units in the tensor of this layer
            n_active: number of active units <= n_units
            batch_size: number of samples to be fed to this layer
            dtype: type of tensor values
            name: name for the tensor
        """
        if n_active is not None and n_active >= n_units:
            raise ValueError("n_active must be < n_units")

        shape = [batch_size, n_units]
        super().__init__(n_units, shape, dtype, name)

        # if n_active is not None convert to SparseTensor
        if n_active is None:
            self.placeholder = array_ops.placeholder(dtype=self.dtype, shape=self.shape, name=self.name)
            self.tensor = self.placeholder
        else:
            self.placeholder = array_ops.placeholder(dtype=self.dtype, shape=[batch_size, n_active], name=self.name)
            self.tensor = transform.flat_indices_to_sparse_tensor(self.placeholder, self.shape)


class SparseInput(Layer):
    """ Sparse Input Layer.

    Creates an op that depends on a `sparse_placeholder` to which one must feed a ``SparseTensorValue``.

    Notes:
        ``SparseTensorValues`` can be created with empty values, but this layer will require the number of values
        to be the same as the number of indices. If this is not the case, an ``InvalidArgumentError`` will be thrown
        when the `TensorFlow` graph is evaluated.
    """

    def __init__(self, n_units, batch_size=None, dtype=dtypes.float32, name="sparse_input"):
        """

        Args:
            n_units (int): the number of output units for this layer
            batch_size (int): batch_size for the input, helps to define the shape for this sparse layer
            dtype: the output type for the values in the ``SparseTensor`` that this layer outputs
            name: name for the layer
        """
        shape = [batch_size, n_units]
        super().__init__(n_units, shape, dtype, name)

        with ops.name_scope(name):
            self.placeholder = array_ops.sparse_placeholder(dtype, self.shape, name)

            n_indices = array_ops.shape(self.placeholder.indices)[0]
            n_values = array_ops.shape(self.placeholder.values)[0]

            valid_values = check_ops.assert_equal(n_indices, n_values, message="Invalid number of values")
            with ops.control_dependencies([valid_values]):
                values = array_ops.identity(self.placeholder.values)

            self.tensor = SparseTensor(self.placeholder.indices, values, self.placeholder.dense_shape)


class Linear(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 init=random_uniform,
                 weights=None,
                 bias=False,
                 dtype=dtypes.float32,
                 name="linear"):

        shape = [layer.shape[1], n_units]
        super().__init__(n_units, shape=shape,
                         dtype=dtype,
                         name=name)

        # if weights are passed, check that their shape matches the layer shape
        if weights is not None:
            (_, s) = weights.get_shape()
            if s != n_units:
                raise ValueError("shape mismatch: layer expects (,{}), weights have (,{})".format(n_units, s))

        with vscope.variable_scope(name):
            # init weights
            if weights is not None:
                self.weights = weights
            else:
                self.weights = vscope.get_variable("w", initializer=init(self.shape))

            # y = xW
            if layer.is_sparse():
                sp_values = layer.tensor
                sp_indices = transform.sp_indices_from_sp_values(sp_values)

                lookup_sum = embedding_lookup_sparse(params=self.weights,
                                                     sp_ids=sp_indices,
                                                     sp_weights=sp_values,
                                                     combiner="sum",
                                                     name=self.name + "_embeddings")
                self.tensor = lookup_sum
            else:
                self.tensor = math_ops.matmul(layer.tensor, self.weights)

            # y = xW + [b]
            if bias:
                self.bias = vscope.get_variable("b", initializer=array_ops.zeros([self.n_units]))
                self.tensor = bias_add(self.tensor, self.bias, name="a")


class ToSparse(Layer):
    """ Transforms the previous layer into a sparse layer.

    This means that the current layer.tensor is a ``SparseTensor``
    """

    def __init__(self, layer):
        super().__init__(layer.n_units, layer.shape, layer.dtype, layer.name + "_sparse")

        with name_scope(self.name):
            if layer.is_sparse():
                self._forward(layer)
            else:
                self.tensor = transform.to_sparse(layer.tensor)


class ToDense(Layer):
    """ Transforms the previous layer into a dense layer.

    """

    def __init__(self, layer):
        super().__init__(layer.n_units, layer.shape, layer.dtype, layer.name + "_dense")

        with name_scope(self.name):
            if layer.is_sparse():
                self.tensor = transform.to_dense(layer.tensor)
            else:
                self._forward(layer)


class Dropout(Layer):
    def __init__(self, layer, keep_prob=0.2, seed=None):
        super().__init__(layer.n_units, layer.shape, layer.dtype, layer.name + "_dropout")

        self.seed = seed
        self.keep_prob = keep_prob

        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=dtypes.float32, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if keep_prob:
            self._forward(layer)
        else:
            if layer.is_sparse():
                self.tensor = transform.sparse_dropout(layer.tensor, keep_prob, seed)
            else:
                self.tensor = dropout(layer.tensor, self.keep_prob, seed=seed)


class GaussianNoise(Layer):
    def __init__(self, layer, mean=0.0, stddev=0.2, seed=None):
        super().__init__(n_units=layer.n_units,
                         shape=layer.shape,
                         dtype=layer.dtype,
                         name=layer.name + "_gaussian_noise")

        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        with name_scope(self.name):
            if layer.is_sparse():
                self.tensor = transform.to_dense(self.tensor)
            else:
                self.tensor = layer.tensor

            noise_shape = array_ops.shape(self.tensor)
            noise = random_ops.random_normal(noise_shape, mean, stddev, seed=seed, dtype=dtypes.float32)
            self.tensor = math_ops.add(self.output, noise)


class SparseGaussianNoise(Layer):
    def __init__(self, layer, density=0.1, mean=0.0, stddev=0.2, dtype=None, seed=None):
        self.dtype = layer.dtype
        if dtype is not None:
            self.dtype = dtype
        super().__init__(n_units=layer.n_units,
                         shape=layer.shape,
                         dtype=self.dtype,
                         name=layer.name + "_sparse_gaussian_noise")
        self.density = density
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        with name_scope(self.name):
            noise_shape = layer.shape
            noise = sparse_random_normal(noise_shape, density, mean, stddev, self.dtype, seed)

            if layer.is_sparse():
                self.tensor = transform.sparse_put(layer.tensor, noise)
            else:
                self.tensor = transform.dense_put(layer.tensor, noise)


class SaltPepperNoise(Layer):
    """ Adds destructive Salt&Pepper noise to the previous layer.

    Generates a random salt&pepper mask that will corrupt a given density of the previous layer.
    It always generates a symmetrical noise mask, meaning that it corrupts

    layer.shape[1].value * density // 2
    with salt, and the same amount with pepper

    if the proportion amounts to less than 2 entries, the previous layer is simply forwarded
    """

    def __init__(self, layer, density=0.1, salt_value=1, pepper_value=-1, seed=None):
        super().__init__(layer.n_units, layer.shape, layer.dtype, layer.name + "_sp_noise")

        self.density = density
        self.seed = seed
        self.salt_value = salt_value
        self.pepper_value = pepper_value

        # do nothing if amount of noise is 0
        if density == 0.0 or self.num_corrupted() > 0:
            self._forward(layer)
        else:
            noise_shape = layer.shape

            if noise_shape[0] is None:
                batch_size = array_ops.shape(layer.tensor, out_type=dtypes.int64)[0]
            else:
                batch_size = noise_shape[0]

            noise_shape = [batch_size, noise_shape[1]]
            noise = salt_pepper_noise(noise_shape, density, salt_value, pepper_value, seed)

            if layer.is_sparse(layer):
                self.tensor = transform.sparse_put(layer.tensor, noise)
            else:
                self.tensor = transform.dense_put(layer.tensor, noise)

    def num_corrupted(self):
        """ Returns the number of entries corrupted by noise per sample"""
        noise_shape = self.shape
        num_noise = int(self.density * noise_shape[1])

        if num_noise < 2:
            num_noise = 0
        else:
            num_noise = (num_noise // 2) * 2
        return num_noise


class Activation(Layer):
    def __init__(self, layer, fn=array_ops.identity):
        super().__init__(layer.n_units, layer.shape, layer.dtype, layer.name + "_activation")
        self.fn = fn
        self.output = self.fn(layer.tensor, name=self.name)


class Bias(Layer):
    """ Bias Layer

    A simple way to add a bias to a given layer, the dimensions of this variable
    are determined by the given layer and it is initialised with zeros
    """

    def __init__(self, layer, name="bias"):
        bias_name = layer.dtype, "{}_{}".format(layer.name, name)
        super().__init__(layer.n_units, layer.shape, bias_name)

        with vscope.variable_scope(self.name):
            self.bias = vscope.get_variable("b", initializer=array_ops.zeros([self.n_units]))
            self.tensor = bias_add(layer.tensor, self.bias, name="tensor")


class Merge(Layer):
    """Merge Layer

    Merges a list layers by combining their tensors with a merging function.
    Allows for the tensor of each layer to be weighted.

    This is just a container that for convenience takes the tensor of each given layer (which is generaly a tensor),
    and applies a merging function.
    """

    def __init__(self,
                 layers,
                 weights=None,
                 merge_fn=math_ops.add_n,
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

        super().__init__(layers[0].n_units, layers[0].shape, layers[0].dtype, name)

        with name_scope(name):
            if weights is not None:
                for i in range(len(layers)):
                    layers[i] = math_ops.scalar_mul(weights[i], layers[i].tensor)

            self.output = merge_fn(layers)
