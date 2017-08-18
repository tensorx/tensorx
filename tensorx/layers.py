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

from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vscope
from tensorflow.python.framework.ops import name_scope

from tensorflow.python.ops import random_ops
from tensorflow.python.ops.nn import embedding_lookup, embedding_lookup_sparse, bias_add, dropout
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.init import random_uniform
from tensorx.random import salt_pepper_noise
import tensorx.transform as transform


class Layer:
    def __init__(self, n_units, shape=None, dense_shape=None, dtype=dtypes.float32, name="layer"):
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
            self.dense_shape = self.shape
        else:
            if dense_shape[1] < n_units:
                raise Exception("Shape mismatch: dense_shape[1] < n_units")
            elif dense_shape[0] != self.shape[0]:
                raise Exception("Shape mismatch: dense_shape[0] != self.shape[0]")
            else:
                self.dense_shape = dense_shape

        # convert shapes to TensorShapes
        # this allows us to init shapes with lists and tuples with None
        self.shape = TensorShape(self.shape)
        self.dense_shape = TensorShape(self.dense_shape)

        # has an output (tensor) attribute
        self.output = None


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
            n_units: number of units in the output of this layer
            n_active: number of active units <= n_units
            batch_size: number of samples to be fed to this layer
            dtype: type of tensor values
            name: name for the tensor
        """
        if n_active is not None and n_active >= n_units:
            raise ValueError("n_active must be < n_units")

        dense_shape = [batch_size, n_units]

        if n_active is not None:
            if dtype != dtypes.int64:
                raise TypeError("If n_active is not None, dtype must be set to dt.int64")
            shape = [batch_size, n_active]
        else:
            shape = [batch_size, n_units]

        super().__init__(n_units, shape, dense_shape, dtype, name)

        self.output = array_ops.placeholder(dtype=self.dtype, shape=self.shape, name=self.name)
        self.key = self.output


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

    def __init__(self, n_units, n_active, values=False, batch_size=None, dtype=dtypes.float32, name="sparse_input"):
        shape = [batch_size, n_active]
        dense_shape = [batch_size, n_units]
        super().__init__(n_units, shape, dense_shape, dtype, name)

        self.n_active = n_active
        self.values = values

        with ops.name_scope(name):
            self.sp_indices = array_ops.sparse_placeholder(dtypes.int64, dense_shape, name)

            if values:
                self.sp_values = array_ops.sparse_placeholder(dtype, dense_shape, name=name + "_values")
            else:
                self.sp_values = None

            self.output = self.sp_indices, self.sp_values

            # key is just an alias for output for readability purposes when using this with Tensorflow run
            self.key = self.output


class Linear(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 init=random_uniform,
                 weights=None,
                 bias=False,
                 dtype=dtypes.float32,
                 name="linear"):

        shape = [layer.dense_shape[1], n_units]
        super().__init__(n_units, shape=shape,
                         dense_shape=shape,
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
            if hasattr(layer, "sp_indices"):
                indices = getattr(layer, "sp_indices")
                values = getattr(layer, "sp_values", None)

                lookup_sum = embedding_lookup_sparse(params=self.weights,
                                                     sp_ids=indices,
                                                     sp_weights=values,
                                                     combiner="sum",
                                                     name=self.name + "_embeddings")
                self.output = lookup_sum
            else:
                if layer.shape.is_compatible_with(layer.dense_shape):
                    self.output = math_ops.matmul(layer.output, self.weights)
                else:
                    lookup = embedding_lookup(params=self.weights,
                                              ids=layer.output,
                                              name=self.name + "_embeddings")
                    self.output = math_ops.reduce_sum(lookup, axis=1)

            # y = xW + [b]
            if bias:
                self.bias = vscope.get_variable("b", initializer=array_ops.zeros([self.n_units]))
                self.output = bias_add(self.output, self.bias, name="a")


class ToSparse(Layer):
    """ Transforms the previous layer into a sparse representation

    meaning the current layer will provide the following attributes:
        sp_indices
        sp_values
    """

    def __init__(self, layer):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_sparse")

        with name_scope(self.name):
            if hasattr(layer, "sp_indices"):
                """sparse layer - forward"""
                self.sp_indices = layer.sp_indices
                if layer.sp_values is None:
                    self.sp_values = transform.default_sp_values(self.sp_indices)
                else:
                    self.sp_values = layer.sp_values
            elif not layer.shape.is_compatible_with(layer.dense_shape):
                """flat index sparse layer"""
                dim = layer.dense_shape.as_list()[1]
                # possibly unknown at __init__ time, so compute it
                batch_size = array_ops.shape(layer.output)[0]
                dense_shape = [batch_size, dim]
                dense_shape = math_ops.cast(dense_shape, dtypes.int64)

                self.sp_indices = transform.flat_indices_to_sparse(layer.output, dense_shape)
                self.sp_values = transform.default_sp_values(self.sp_indices)
            else:
                """dense Layer"""
                self.sp_indices, self.sp_values = transform.to_sparse(layer.output)

            self.output = self.sp_indices, self.sp_values


def _is_sparse_layer(layer):
    return hasattr(layer, "sp_indices") or (
        not layer.shape.is_compatible_with(layer.dense_shape) and layer.dtype == dtypes.int64)


def _sparse_layer_to_dense_tensor(layer):
    dense = None
    if hasattr(layer, "sp_indices"):
        """Converts Sparse Layer to Dense """
        dense = transform.to_dense(layer.sp_indices, layer.sp_values)
    elif not layer.shape.is_compatible_with(layer.dense_shape) and layer.dtype == dtypes.int64:
        """Converts Sparse Index Layer to Dense """
        dense = transform.flat_indices_to_dense(layer.output, layer.dense_shape)

    return dense


class ToDense(Layer):
    """ Transforms the previous layer into a dense representation

        meaning the input layer provides:
            sp_indices
            [sp_values] if sp_values is None, the values are set to 1.0
    """

    def __init__(self, layer):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_sparse")

        with name_scope(self.name):
            self.output = _sparse_layer_to_dense_tensor(layer)


class Dropout(Layer):
    def __init__(self, layer, keep_prob, seed):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_dropout")

        self.seed = seed
        self.keep_prob = keep_prob

        if not _is_sparse_layer(layer):
            """Apply dropout to Dense Layer"""
            self.output = dropout(layer.output, keep_prob=keep_prob)
        else:
            if hasattr(layer, "sp_indices"):
                """Apply dropout to SparseLayer
                    by applying dropout to values tensor
                """
                sp_indices = layer.sp_indices
                sp_values = layer.sp_values

                if sp_values is None:
                    default_values = array_ops.constant(1.0, dtypes.float32, shape=array_ops.shape(sp_indices.values))

                    drop_values = dropout(default_values, keep_prob, seed=seed)
                    not_zero = math_ops.not_equal(drop_values, 0)
                    not_zero_indices = array_ops.where(not_zero)
                    values = array_ops.boolean_mask(drop_values, not_zero)

                    indices = array_ops.gather(sp_indices, not_zero_indices)
                    _, flat_indices = array_ops.unstack(indices, axis=-1)
                    sp_indices = SparseTensor(indices, flat_indices, self.dense_shape)
                    sp_values = SparseTensor(indices, values, self.dense_shape)

                self.sp_indices = sp_indices
                self.sp_values = sp_values

                self.output = self.sp_indices, self.sp_values

            elif not layer.shape.is_compatible_with(layer.dense_shape) and layer.dtype == dtypes.int64:
                indices = transform.enum_row(layer.output)

                default_values = array_ops.constant(1.0, dtypes.float32, shape=[layer.shape[0]])
                drop_values = dropout(default_values, keep_prob, seed=seed)
                not_zero = math_ops.not_equal(drop_values, 0)
                not_zero_indices = array_ops.where(not_zero)
                values = array_ops.boolean_mask(drop_values, not_zero)

                indices = array_ops.gather(indices, not_zero_indices)
                _, flat_indices = array_ops.unstack(indices, axis=-1)
                sp_indices = SparseTensor(indices, flat_indices, self.dense_shape)
                sp_values = SparseTensor(indices, values, self.dense_shape)

                self.sp_indices = sp_indices
                self.sp_values = sp_values

                self.output = self.sp_indices, self.sp_values
            else:
                raise TypeError("Invalid Layer, could not be corrupted with dropout")


class GaussianNoise(Layer):
    def __init__(self, layer, noise_amount=0.1, mean=0.0, stddev=0.2, seed=None):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_gaussian_noise")

        self.noise_amount = noise_amount
        self.stddev = stddev
        self.seed = seed

        # do nothing if amount of noise is 0
        if noise_amount == 0.0:
            self.output = layer.output
        else:
            if _is_sparse_layer(layer):
                self.output = _sparse_layer_to_dense_tensor(layer)

            noise = random_ops.random_normal(array_ops.shape(self.output), mean, stddev, seed=seed,
                                             dtype=dtypes.float32)
            self.output = math_ops.add(self.output, noise)


class SaltPepperNoise(Layer):
    def __init__(self, layer, noise_amount=0.1, max_value=1, min_value=0, seed=None):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_sp_noise")

        self.noise_amount = noise_amount
        self.seed = seed

        # do nothing if amount of noise is 0
        if noise_amount == 0.0:
            self.output = layer.output
        else:
            num_noise = int(layer.n_units * noise_amount)
            batch_size = self.shape[0]

            if hasattr(layer, "sp_indices"):
                """corrupt sparse layer"""
                sp_indices = getattr(layer, "sp_indices")
                sp_values = getattr(layer, "sp_values", default=None)

                if sp_values is None:
                    values = array_ops.constant(1.0, dtypes.float32, shape=array_ops.shape(sp_indices.values))
                    indices = sp_indices.indices
                    sp_values = SparseTensor(indices, values)

                noise = salt_pepper_noise(layer.dense_shape, noise_amount, max_value, min_value, seed)
                sp_values = transform.sparse_put(sp_values, noise)

                indices = sp_values.indices
                _, flat_indices = array_ops.unstack(indices, axis=-1)
                sp_indices = SparseTensor(indices, flat_indices, sp_values.dense_shape)

                self.sp_indices = sp_indices
                self.sp_values = sp_values

                self.output = self.sp_indices, self.sp_values
            elif not layer.shape.is_compatible_with(layer.dense_shape) and layer.dtype == dtypes.int64:
                """corrupt sparse index layer
                converts the original layer to a sparse layer
                the input layer is a batch of flat indices: the indices correspond to each sample, not to the indices
                of a sparse matrix
                """
                indices = transform.enum_row(layer.output)
                values = array_ops.constant(1.0, dtypes.float32, shape=[array_ops.shape(indices)[0]])
                dense_shape = layer.dense_shape
                sp_values = SparseTensor(indices, values, dense_shape)

                noise = salt_pepper_noise(dense_shape, noise_amount, max_value, min_value, seed)
                sp_values = transform.sparse_put(sp_values, noise)

                flat_indices = array_ops.reshape(layer.output, [-1])
                sp_indices = SparseTensor(indices, flat_indices, dense_shape)

                self.sp_indices = sp_indices
                self.sp_values = sp_values

                self.output = self.sp_indices, self.sp_values

            elif layer.shape.is_compatible_with(layer.dense_shape):
                """corrupt dense layer"""
                noise = salt_pepper_noise(layer.dense_shape, noise_amount, max_value, min_value, seed)
                self.output = transform.dense_put(layer.output, noise)
            else:
                raise ValueError("Invalid Layer Error: could not be corrupted")


class Activation(Layer):
    def __init__(self, layer, fn=array_ops.identity):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_activation")
        self.fn = fn
        self.output = self.fn(layer.output, name=self.name)


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
            self.output = bias_add(layer.tensor, self.bias, name="output")


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

        super().__init__(layers[0].n_units, layers[0].shape, layers[0].dense_shape, layers[0].dtype, name)

        with name_scope(name):
            if weights is not None:
                for i in range(len(layers)):
                    layers[i] = math_ops.scalar_mul(weights[i], layers[i].output)

            self.output = merge_fn(layers)
