""" Neural Network Layers

All layers contain a certain number of units, its shape, name and a tensor member
which gives us a handle for a TensorFlow tensor that can be evaluated.

Types of layers:

    * **Input**: wrap around TensorFlow placeholders, ``input_layers`` is empty.
    * **Dense**:  a layer that outputs a tensorflow ``Tensor``.
    * **Sparse**: a layer that outputs a tensorflow ``SparseTensor``.

"""
from functools import partial
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework.ops import name_scope

from tensorflow.python.ops import random_ops, sparse_ops
from tensorflow.python.ops.nn import embedding_lookup_sparse, bias_add, dropout, embedding_lookup
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.init import random_uniform, zero_init
from tensorx.random import salt_pepper_noise, sparse_random_normal
from tensorx import transform
from tensorx import utils as txutils
from tensorx.metrics import pairwise_cosine_distance


def _as_list(elems):
    """ returns a list from the given element(s)

    Args:
        elems: one or more objects

    Returns:
        a list with the elements in elems
    """
    if elems is None:
        elems = []
    elif isinstance(elems, (list, tuple)):
        elems = list(elems)
    else:
        elems = [elems]
    return elems


def layers_to_list(output_layers):
    """ Converts a layer or a list of layers to a list of all the layers connected to it.

    Warning:
        should be used with the last layer, the output layer from which the list is built following the references
        to input layers for each layer.

    Args:
        output_layers: the layer from which we wish to build the list of layers involved in the computation

    Returns:
        :obj:`list` of :class:`Layer`: a list of unique layers involved in the computation ordered from input to output
        in a breadth-first fashion.

    """
    flat_layers, visited, stack = [], set(), _as_list(output_layers)
    while stack:
        layer = stack.pop(0)
        if layer not in visited:
            visited.add(layer)
            flat_layers.append(layer)
            next_layers = layer.input_layers
            if len(next_layers) > 1:
                next_layers.reverse()
            stack.extend(layer.input_layers)

    flat_layers.reverse()
    return flat_layers


class Layer:
    """ Layer.

    Attributes:
        input_layers: a list of Layers that serve as input to the current layer
        n_units: the number of units for the current layer
        tensor: a ``Tensor`` or ``SparseTensor`` if the layer is dense or sparse respectively
        dtype: the tensorflow dtype for the output tensor
        name: a name used to build a tensorflow named_scope for the layer
        variable_names: a list of `tf.Variable` names that get be get with get_variable without a scope




    Args:
        input_layers: a single layer,a list of input layers, or None if no inputs are required
        n_units: dimension of input vector (dimension of columns in case batch_size != None
        shape: [batch size, input dimension]
        dtype: expected input TensorFlow data type
        name: layer name (used to nam the placeholder)



    """

    def __init__(self, input_layers, n_units, shape=None, dtype=dtypes.float32, name="layer"):

        self.n_units = n_units
        self.name = name
        self.dtype = dtype
        self.input_layers = _as_list(input_layers)

        # stores the variables if this layer has any
        self.variable_names = []

        if shape is None:
            self.shape = [None, n_units]
        else:
            if shape[1] < n_units:
                raise ValueError("Shape mismatch: shape[1] < n_units")
            self.shape = shape

        # has an tensor (tensor) attribute
        self.tensor = None

    def _add_variable(self, var):
        if not isinstance(var, variables.Variable):
            raise TypeError("Expected a tf.Variable got {t} instead".format(t=type(var)))
        self.variable_names.append(var.op.name)

    def is_sparse(self):
        """ Checks if the current layer is sparse

        A layer is sparse if its output tensor is a ``SparseTensor``, it is dense if the output tensor is a ``Tensor``.

        Returns:
            ``bool``: returns True if the current layer is sparse, False otherwise.

        """
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

    def __str__(self):
        """ Informal string representation for a layer consists of Layer Class name, number of units and if its
        Sparse or Dense.

        Returns:
            :obj:`str`: a :obj:`str` with the informal representation for a layer instance.

        """
        class_name = type(self).__name__
        sparse_dense = "[Sparse]" if self.is_sparse() else "[Dense]"
        return "{class_name}({n_units},{dtype}){sparse_dense}".format(class_name=class_name,
                                                                      n_units=self.n_units,
                                                                      dtype=self.dtype,
                                                                      sparse_dense=sparse_dense)


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
        super().__init__(None, n_units, shape, dtype, name)

        # if n_active is not None convert to SparseTensor
        if n_active is None:
            self.placeholder = array_ops.placeholder(dtype=self.dtype, shape=self.shape, name=self.name)
            self.tensor = self.placeholder
        else:
            self.n_active = n_active
            self.placeholder = array_ops.placeholder(dtype=self.dtype, shape=[batch_size, n_active], name=self.name)

            indices_shape = txutils.complete_shape(self.placeholder)
            dense_shape = [indices_shape[0], self.shape[1]]
            self.tensor = transform.sparse_one_hot(self.placeholder, dense_shape, dtype=self.dtype)

    def __str__(self):
        class_name = type(self).__name__
        if self.is_sparse():
            str = "{class_name}({n_active}/{n_units},{dtype})[Sparse]".format(class_name=class_name,
                                                                              n_active=self.n_active,
                                                                              n_units=self.n_units,
                                                                              dtype=self.dtype)
        else:
            str = "{class_name}({n_units},{dtype})[Dense]".format(class_name=class_name, n_units=self.n_units,
                                                                  dtype=self.dtype)
        return str


class TensorInput(Layer):
    """ Tensor Input Layer

    Creates a layer from a given tensor that one can then integrate with other layers
    """

    def __init__(self, tensor, n_units, batch_size=None, dtype=dtypes.float32, name="tensor_input"):
        tensor = txutils.to_tensor_cast(tensor, dtype)
        shape = [batch_size, n_units]

        try:
            assert (tensor.get_shape().as_list()[1] == n_units)
            if batch_size is not None:
                assert (tensor.get_shape().as_list()[0] == batch_size)
        except AssertionError:
            raise ValueError(
                "Tensor shape {shape} does not match [batch_size, n_units] = [{batch_size}, {n_units}]".format(
                    shape=tensor.get_shape().as_list(),
                    n_units=n_units,
                    batch_size=batch_size
                ))

        super().__init__(None, n_units, shape, dtype, name)

        self.tensor = tensor


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
        super().__init__(None, n_units, shape, dtype, name)

        with ops.name_scope(name):
            self.placeholder = array_ops.sparse_placeholder(dtype, self.shape, name)
            n_indices = array_ops.shape(self.placeholder.indices)[0]
            n_values = array_ops.shape(self.placeholder.values)[0]

            valid_values = check_ops.assert_equal(n_indices, n_values, message="Invalid number of values")
            with ops.control_dependencies([valid_values]):
                values = array_ops.identity(self.placeholder.values)

            self.tensor = SparseTensor(self.placeholder.indices, values, self.placeholder.dense_shape)


class Linear(Layer):
    """ Linear Layer.

    A fully connected layer with a given number of units.

    Linear transformation Layer creates a linear transformation of the form ``xW + b`` where:

     * ``x``: is a tensor coming from the layer this layer takes as input
     * ``W``: is a variable with trainable weights created by this Layer
     * ``b``: is an optional variable containing biases to be added to the linear transformation

    the transformation becomes an **affine transformation**.

    Note:
        There is no need for **embedding** layers in this library. If the Linear layer receives a sparse layer as input
        it uses the embeddings op, if not, it uses the default multiplication.

    Args:
        layer: an input :class:`Layer` used to build a fully connected layer
        n_units: an :obj:`int` with the number of units for this layer
        init: an initializer for the weights of this layer
        weights: None if we wish the layer to create a new variable or a tensorflow variable otherwise
        bias: if true creates a bias variable and the transformation becomes an affine transformation
        dtype: dtype for the output of this layer
        name: name to create a tensorflow named_scope
    """

    def __init__(self,
                 layer,
                 n_units,
                 init=random_uniform(),
                 weights=None,
                 bias=True,
                 dtype=dtypes.float32,
                 name="linear"):

        shape = [layer.shape[1], n_units]
        super().__init__(layer, n_units, shape, dtype, name)

        # if weights are passed, check that their shape matches the layer shape
        if weights is not None:
            (_, s) = weights.get_shape()
            if s != n_units:
                raise ValueError("shape mismatch: layer expects (,{}), weights have (,{})".format(n_units, s))

        with name_scope(name) as scope, variable_scope.variable_scope(scope):
            # init weights
            if weights is not None:
                self.weights = weights
            else:
                self.weights = variable_scope.get_variable("w",
                                                           shape=self.shape,
                                                           dtype=self.dtype,
                                                           initializer=init)

            self._add_variable(self.weights)
            # y = xW
            if layer.is_sparse():
                sp_values = layer.tensor
                sp_indices = transform.sp_indices_from_sp_tensor(sp_values)

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
                self.bias = variable_scope.get_variable("b",
                                                        shape=[self.n_units],
                                                        dtype=self.dtype,
                                                        initializer=zero_init())
                self._add_variable(self.bias)
                self.tensor = bias_add(self.tensor, self.bias, name="a")


class Lookup(Layer):
    """
    Note:
        If the input is a ``SparseInput`` it expects on element of an n-sequence per row of the ``SparseTensor``
        this is because of how embedding lookup works. Since this layer requires us to supply an exact batch_size
        it will aggregate the final result according to this batch.

        If we want to lookup a batch of sequences of 4 elements, the ``SparseInput`` must have the shape
        [4*2,m] where m is the n_features.

        If the input is ``Input``, for a sequence of 4 elements and batch size of 2, the shape should be [2,4].

    Returns:
        A ``Tensor`` with shape [batch_size,seq_size*n_features] with the features of all elements in the sequence concatenated

    Args:
        input_layer: an ``Input`` layer or ``SparseInput`` layers.
        seq_size: size of the sequence to be looked-up
        feature_shape: lookup table feature dimension
        batch_size: number of sequences to be looked up

    """

    # TODO adaptive feature shape based on input if input has n_active
    def __init__(self,
                 input_layer,
                 seq_size,
                 feature_shape,
                 batch_size,
                 init=random_uniform(),
                 weights=None,
                 dtype=dtypes.float32,
                 name="seq_lookup"):

        self.weight_shape = feature_shape
        n_units = seq_size * feature_shape[1]
        self.batch_size = batch_size
        shape = [batch_size, n_units]

        super().__init__(input_layer, n_units, shape, dtype, name)

        # if weights are passed, check that their shape matches the layer shape
        if weights is not None:
            (_, s) = weights.get_shape()
            if s != feature_shape[1]:
                raise ValueError("shape mismatch: layer expects (,{}), weights have (,{})".format(n_units, s))

        with name_scope(name) as scope, variable_scope.variable_scope(scope):
            # init weights
            if weights is not None:
                self.weights = weights
            else:
                self.weights = variable_scope.get_variable("w", shape=self.weight_shape, initializer=init)

            # y = xW
            if input_layer.is_sparse():
                sp_values = input_layer.tensor
                sp_indices = transform.sp_indices_from_sp_tensor(sp_values)

                # sums the lookups for the same row
                lookup_sum = embedding_lookup_sparse(params=self.weights,
                                                     sp_ids=sp_indices,
                                                     sp_weights=sp_values,
                                                     combiner="sum",
                                                     name=self.name + "_embeddings")

                self.tensor = array_ops.reshape(lookup_sum, [self.batch_size, -1])
            else:
                lookup = embedding_lookup(params=self.weights,
                                          ids=input_layer.tensor)
                self.tensor = array_ops.reshape(lookup, [self.batch_size, -1])


class ToSparse(Layer):
    """ Transforms the previous layer into a sparse layer.

    This means that the current layer.tensor is a ``SparseTensor``
    """

    def __init__(self, layer):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_sparse")

        with name_scope(self.name):
            if layer.is_sparse():
                self._forward(layer)
            else:
                self.tensor = transform.to_sparse(layer.tensor)


class ToDense(Layer):
    """ ToDense transformation layer

    Transforms the previous layer into a dense layer (outputting a dense tensor)
    if the previous layer is already a dense layer, forwards the previous layer doing nothing

    """

    def __init__(self, layer):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_dense")

        with name_scope(self.name):
            if layer.is_sparse():
                self.tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                self._forward(layer)


class Dropout(Layer):
    """ A Dropout Layer that applies the tensorflow dropout op to a given layer.

    With probability ``keep_prob``, outputs the input elements scaled up by ``1 / keep_prob``, otherwise
    outputs ``0``. The scaling is to that the expected sum of the input elements is unchanged.

    Dropout can be viewed a stochastic version of model averaging and prevents the nodes from co-adapting too much. This
    reduces generalisation error during training.

    References:
        [1] "Dropout:  A Simple Way to Prevent Neural Networks from Overfitting"
        http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

    Note:
        Contrary to the tensorflow operator, this layer also works with sparse layers as input it uses:

            * `dropout` from tensorflow for dense layers
            * :class:`tensorx.transform.sparse_dropout` from for sparse layers

    Args:
            layer: an input layer :class:`Layer` to which dropout will be applied
            keep_prob: a scalar float with the probability that each element is kept.
            seed: A Python integer. Used to create a random seed for the dropout op.
    """

    def __init__(self, layer, keep_prob=0.2, seed=None):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_dropout")

        self.seed = seed
        self.keep_prob = keep_prob

        if keep_prob == 0:
            self._forward(layer)
        else:
            with name_scope(self.name):
                if layer.is_sparse():
                    self.tensor = transform.sparse_dropout(layer.tensor, self.keep_prob, seed)
                else:
                    self.tensor = dropout(layer.tensor, self.keep_prob, seed=seed)


class GaussianNoise(Layer):
    """ Gaussian Noise Layer.

    Applies additive gaussian noise to the input layer. If the noise is zero-centered it does not
    change the expected sum of input elements.


    Use Cases:
        This is useful to mitigate overfitting, making a network model more robust. Noise is usually added to inputs,
        activation functions, or to network weights.

    Args:
        layer: an input :class:`Layer` used to build a fully connected layer
        mean: the mean for the gaussian distribution
        stddev: the standard deviation parameter for the gaussian distribution
        seed: A Python integer. Used to create a random seed for the distribution

    """

    def __init__(self, layer, mean=0.0, stddev=0.2, seed=None):
        super().__init__(input_layers=layer,
                         n_units=layer.n_units,
                         shape=layer.shape,
                         dtype=layer.dtype,
                         name=layer.name + "_gaussian_noise")

        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        with name_scope(self.name):
            if layer.is_sparse():
                self.tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                self.tensor = layer.tensor

            noise_shape = array_ops.shape(self.tensor)
            noise = random_ops.random_normal(noise_shape, mean, stddev, seed=seed, dtype=dtypes.float32)

            self.tensor = math_ops.cast(self.tensor, dtypes.float32)
            self.tensor = math_ops.add(self.tensor, noise)


class SparseGaussianNoise(Layer):
    """ Sparse Gaussian Noise Layer

    Applies additive gaussian noise to a given proportion of elements from an input layer.
    If ``density == 1.0``, this is equivalent to :class:`GaussianNoise`.

    """

    def __init__(self, layer, density=0.1, mean=0.0, stddev=0.2, dtype=None, seed=None):
        self.dtype = layer.dtype
        if dtype is not None:
            self.dtype = dtype
        super().__init__(input_layers=layer,
                         n_units=layer.n_units,
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

    Args:
        layer: the input :class:`Layer`
        density: a float scalar with the proportion of elements to be corrupted
        salt_value: the value to be set for the "salt noise" (usually the max value of 1.)
        pepper_value: the value to be set for the "pepper noise" (usually the min value of -1 or 0, but -1 keeps the
        expected sum of the input elements the same.
        seed: A Python integer. Used to create a random seed for the distribution
    """

    def __init__(self, layer, density=0.1, salt_value=1, pepper_value=-1, seed=None):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_sp_noise")

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
    """ Activation Layer

    A container that applies a given function the the tensor of its input layer.
    If the layer is sparse, it converts it to a dense layer.

    You can pass positinoal arguments and keyword arguments for the given function,
    their application works like :func:`functools.partial`.

    Note:
        Most activation functions do not produce a sparse output so no effort is made to adapt existing activation
        functions to sparse input layers. If we know a non-linearity produces a sparse output with high-probability,
        we can use a sparse transformation explicitly with a :class:`ToSparse` layer. Perhaps in the future I can
        include a layer that does this transformation based on a desired sparsity threshold.

    Args:
        layer: the input :class:`Layer`
        fn: a function that produces a Tensor and can be called on the tensor produced by the input layer
        **keywords: the keyword arguments for the given function
    """

    def __init__(self, layer, fn=array_ops.identity, **keywords):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_activation")
        self.fn = partial(fn, **keywords)

        with name_scope(self.name):
            if layer.is_sparse():
                tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                tensor = layer.tensor
            self.tensor = self.fn(tensor)


class Bias(Layer):
    """ Bias Layer
    A simple way to add a bias to a given layer, the dimensions of this variable
    are determined by the given layer and it is initialised with zeros

    Warning:
        :class:`Linear` already have a bias option that if checked adds a bias variable
        to the linear transformation. Do not use this on top of a Linear layer if you already
        have a bias defined, you're just adding another bias variable.

    Args:
        layer: the input :class:`Layer`
        name: the name for the bias layer tensorflow scope
    """

    def __init__(self, layer, name="bias"):
        bias_name = layer.dtype, "{}_{}".format(layer.name, name)
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, bias_name)

        with name_scope(name) as scope, variable_scope.variable_scope(scope):
            self.bias = variable_scope.get_variable("b", shape=[self.n_units], initializer=zero_init())
            if layer.is_sparse():
                tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                tensor = layer.tensor
            self.tensor = bias_add(tensor, self.bias, name="tensor")


class Merge(Layer):
    """Merge Layer

    Merges a list layers by combining their tensors with a merging function.
    Allows for the tensor of each layer to be weighted.

    This is just a container that for convenience takes the tensor of each given layer (which is generaly a tensor),
    and applies a merging function.

    Args:
            layers: a list of layers with the same number of units to be merged
            weights: a list of weights
            merge_fn: must operate on a list of tensors
            name: name for layer which creates a named-scope

    Requires:
        * ``len(layers) == len(weights)``
        * all layers must have the same number of units
        * all layers must be of the same type (sparse or dense) and have the same dtype
        * the merge_fn should be applicable to the ``Tensor`` if the layers are dense, and to ``SparseTensor`` otherwise
    """

    def __init__(self,
                 layers,
                 weights=None,
                 merge_fn=math_ops.add_n,
                 name="merge"):
        if len(layers) < 2:
            raise Exception("Expecting a list of layers with len >= 2")

        if weights is not None and len(weights) != len(layers):
            raise Exception("len(weights) must be equals to len(layers)")

        super().__init__(layers, layers[0].n_units, layers[0].shape, layers[0].dtype, name)

        with name_scope(name):
            if weights is not None:
                tensors = [math_ops.scalar_mul(weights[i], layers[i].tensor) for i in range(len(layers))]
            else:
                tensors = [layer.tensor for layer in layers]
            self.tensor = merge_fn(tensors)


class Add(Merge):
    """ Adds the outputs of multiple layers with the same shape

    Args:
            layers: a list of layers with the same number of units to be merged
            weights: a list of weights
            name: name for layer scope
    """

    def __init__(self, layers, weights=None, name="add"):
        super().__init__(layers, weights, math_ops.add_n, name)


class Concat(Layer):
    """ Concat Layer

    Concatenates input layers on the last dimension

    Args:
        layers: a :obj:`list` of :class:`Layer`
        name: name for the layer scope
    """

    def __init__(self, layers, name="concat"):
        first, *rest = layers
        if not all(layer.dtype == first.dtype for layer in rest):
            raise ValueError("Layers must have the same type to be concatenated")

        total_units = sum([layer.n_units for layer in layers])
        super().__init__(layers, total_units, dtype=first.dtype, name=name)

        tensors = [layer.tensor for layer in layers]
        self.tensor = array_ops.concat(tensors, axis=-1)


class SOMLinear(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 map_size,
                 map_distance=pairwise_cosine_distance,
                 init=random_uniform(), weights=None, bias=True, dtype=dtypes.float32, name="som_linear"):
        shape = [layer.shape[1], n_units]
        super().__init__(layer, n_units, shape, dtype, name)

        self.map_size = map_size
        self.weights_shape = [map_size, layer.shape[1], n_units]
        self.map_weights = None
        self.map_shape = [map_size, layer.shape[1]]
        self.bias = None

        with name_scope(name) as scope, variable_scope.variable_scope(scope):
            # init weights
            self.weights = variable_scope.get_variable("w", shape=self.weights_shape, initializer=init)
            self.map_weights = variable_scope.get_variable("som_w", shape=self.map_shape, initializer=init)

            if layer.is_sparse():
                self.tensor = None
            else:
                som_dist = pairwise_cosine_distance(layer.tensor, self.map_weights)
                # Best Matching Unit (BMU)
                bmu = math_ops.argmin(som_dist, axis=0)


__all__ = ["Input",
           "TensorInput",
           "SparseInput",
           "Activation",
           "ToSparse",
           "ToDense",
           "Bias",
           "Linear",
           "Merge",
           "Concat",
           "Add",
           "Dropout",
           "GaussianNoise",
           "SparseGaussianNoise",
           "SaltPepperNoise",
           "Lookup",
           "layers_to_list"
           ]
