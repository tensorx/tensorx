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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vscope
from tensorflow.python.framework.ops import name_scope

from tensorflow.python.ops import random_ops
from tensorflow.python.ops.nn import embedding_lookup, embedding_lookup_sparse, bias_add, dropout
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.init import random_uniform
from tensorx.random import salt_pepper_noise, sparse_random_mask, sparse_random_normal
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
        self.dense_shape = prev_layer.dense_shape
        self.output = prev_layer.output

        if _is_flat_sparse(prev_layer):
            self.dtype = prev_layer.dtype
        elif _is_sparse(prev_layer):
            self.sp_indices = prev_layer.sp_indices
            self.sp_values = prev_layer.sp_values


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
                # auto-completes the shape if batch size is unknown
                self.sp_indices = transform.flat_indices_to_sparse(layer.output, layer.dense_shape)
                self.sp_values = transform.default_sp_values(self.sp_indices)
            else:
                """dense Layer"""
                self.sp_indices, self.sp_values = transform.to_sparse(layer.output)

            self.output = self.sp_indices, self.sp_values


def _is_sparse(layer):
    return hasattr(layer, "sp_indices")


def _is_flat_sparse(layer):
    return not layer.shape.is_compatible_with(layer.dense_shape) and layer.dtype == dtypes.int64


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
    def __init__(self, layer, keep_prob=0.2, seed=None):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_dropout")

        self.seed = seed
        self.keep_prob = keep_prob

        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=dtypes.float32, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if not (_is_sparse(layer) or _is_flat_sparse(layer)):
            """Apply dropout to Dense Layer"""
            # Do nothing if we know keep_prob == 1
            if tensor_util.constant_value(keep_prob) == 1:
                self.output = layer.output
            else:
                self.output = dropout(layer.output, self.keep_prob, seed=seed)
        else:
            if hasattr(layer, "sp_indices"):
                """Apply dropout to SparseLayer
                    by applying dropout to values tensor
                """
                if tensor_util.constant_value(keep_prob) == 1:
                    self.sp_indices, self.sp_values = layer.sp_indices, layer.sp_values
                else:
                    self.sp_indices, self.sp_values = transform.sparse_dropout(layer.sp_indices,
                                                                               layer.sp_values,
                                                                               keep_prob, seed)
                self.output = self.sp_indices, self.sp_values

            elif not layer.shape.is_compatible_with(layer.dense_shape) and layer.dtype == dtypes.int64:
                if tensor_util.constant_value(keep_prob) == 1:
                    self.output = layer.output
                else:
                    sp_indices = transform.flat_indices_to_sparse(layer.output, self.dense_shape)
                    sp_values = transform.default_sp_values(sp_indices)

                    self.sp_indices, self.sp_values = transform.sparse_dropout(sp_indices,
                                                                               sp_values,
                                                                               keep_prob, seed)
                    self.output = self.sp_indices, self.sp_values
            else:
                raise TypeError("Invalid Layer, could not be corrupted with dropout")


class GaussianNoise(Layer):
    def __init__(self, layer, mean=0.0, stddev=0.2, seed=None):
        super().__init__(n_units=layer.n_units,
                         shape=layer.dense_shape,
                         dense_shape=layer.dense_shape,
                         dtype=layer.dtype,
                         name=layer.name + "_gaussian_noise")

        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        with name_scope(self.name):
            if _is_sparse(layer) or _is_flat_sparse(layer):
                self.output = _sparse_layer_to_dense_tensor(layer)
            else:
                self.output = layer.output

            noise = random_ops.random_normal(array_ops.shape(self.output), mean, stddev, seed=seed,
                                             dtype=dtypes.float32)
            self.output = math_ops.add(self.output, noise)


class SparseGaussianNoise(Layer):
    def __init__(self, layer, density=0.1, mean=0.0, stddev=0.2, dtype=None, seed=None):
        self.dtype = layer.dtype
        if dtype is not None:
            self.dtype = dtype
        super().__init__(n_units=layer.n_units,
                         shape=layer.dense_shape,
                         dense_shape=layer.dense_shape,
                         dtype=self.dtype,
                         name=layer.name + "_sparse_gaussian_noise")
        self.density = density
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        with name_scope(self.name):
            if _is_sparse(layer):
                # get sp_values and sp_indices
                sp_indices = layer.sp_indices
                if layer.sp_values is None:
                    sp_values = transform.default_sp_values(sp_indices, self.dtype)
                else:
                    sp_values = layer.sp_values
                    if layer.dtype != self.dtype:
                        sp_values = math_ops.cast(sp_values, self.dtype)

                # corrupt values
                noise_shape = self.sp_indices.dense_shape
                noise = sparse_random_normal(noise_shape, density, mean, stddev, self.dtype, seed)
                self.sp_values = transform.sparse_put(sp_values, noise)
                self.sp_indices = transform.sp_values_to_sp_indices(sp_values)
                self.output = self.sp_indices, self.sp_values
            elif _is_flat_sparse(layer):
                raise NotImplementedError()


class SaltPepperNoise(Layer):
    """ Adds destructive Salt&Pepper noise to the previous layer.

    Generates a random salt&pepper mask that will corrupt a given density of the previous layer.
    It always generates a symmetrical noise mask, meaning that it corrupts

    layer.dense_shape[1].value * density // 2
    with salt, and the same amount with pepper

    if the proportion amounts to less than 2 entries, the previous layer is simply forwarded
    """

    def __init__(self, layer, density=0.1, salt_value=1, pepper_value=-1, seed=None):
        super().__init__(layer.n_units, layer.shape, layer.dense_shape, layer.dtype, layer.name + "_sp_noise")

        self.density = density
        self.seed = seed
        self.salt_value = salt_value
        self.pepper_value = pepper_value

        # do nothing if amount of noise is 0
        if density == 0.0:
            self.output = layer.output
        else:
            dense_shape = layer.dense_shape
            noise_shape = dense_shape.as_list()

            if noise_shape[0] is None:
                batch_size = array_ops.shape(layer.output, out_type=dtypes.int64)[0]
            else:
                batch_size = noise_shape[0]

            noise_shape = [batch_size, noise_shape[1]]

            if self.num_corrupted() > 0:
                noise = salt_pepper_noise(noise_shape, density, salt_value, pepper_value, seed)

            # transform or forward according to the type of the previous layer
            if _is_sparse(layer):
                # CORRUPT SPARSE LAYER
                if self.num_corrupted() > 0:
                    sp_indices = layer.sp_indices
                    sp_values = layer.sp_values

                    # corrupt
                    if sp_values is None:
                        sp_values = transform.default_sp_values(sp_indices)

                    self.sp_values = transform.sparse_put(sp_values, noise)
                    self.sp_indices = transform.sp_values_to_sp_indices(sp_values)
                    self.output = self.sp_indices, self.sp_values
                else:
                    self._forward(layer, self)

            elif _is_flat_sparse(layer):
                # CORRUPT FLAT SPARSE LAYER
                if self.num_corrupted() > 0:
                    self.shape = self.dense_shape
                    # corrupt
                    sp_indices = transform.flat_indices_to_sparse(layer.output)
                    sp_values = transform.default_sp_values(sp_indices)

                    sp_values = transform.sparse_put(sp_values, noise)
                    sp_indices = transform.sp_values_to_sp_indices(sp_values)

                    self.sp_indices = sp_indices
                    self.sp_values = sp_values
                    self.output = self.sp_indices, self.sp_values
                else:
                    self._forward(layer)

            else:
                # CORRUPT DENSE LAYER
                if self.num_corrupted() > 0:
                    self.output = transform.dense_put(layer.output, noise)
                else:
                    self._forward(layer)

    def num_corrupted(self):
        """ Returns the number of entries corrupted by noise per sample"""
        noise_shape = self.dense_shape.as_list()
        num_noise = int(self.density * noise_shape[1])

        if num_noise < 2:
            num_noise = 0
        else:
            num_noise = (num_noise // 2) * 2
        return num_noise


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
