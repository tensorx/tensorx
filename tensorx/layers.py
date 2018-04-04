""" Neural Network Layers

All layers contain a certain number of units, its shape, name and a tensor member
which gives us a handle for a TensorFlow tensor that can be evaluated.

Types of layers:

    * **Input**: wrap around TensorFlow placeholders, ``input_layers`` is empty.
    * **Dense**:  a layer that outputs a tensorflow ``Tensor``.
    * **Sparse**: a layer that outputs a tensorflow ``SparseTensor``.

"""
from functools import partial
import itertools

from tensorflow.python.framework import ops, dtypes

from tensorflow.python.framework.ops import Tensor, name_scope
from tensorflow.python.ops import array_ops, check_ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.variable_scope import _pure_variable_scope as pure_variable_scope
from tensorflow.python.ops import variable_scope

from tensorflow.python.ops import random_ops, sparse_ops
from tensorflow.python.ops.nn import embedding_lookup_sparse, bias_add, dropout, embedding_lookup
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.init import random_uniform, zero_init
from tensorx.random import salt_pepper_noise, sparse_random_normal
from tensorx import transform
from tensorx import utils as txutils
from tensorx.metrics import batch_cosine_distance
from tensorx.activation import sigmoid, elu
from tensorx import math as mathx

from contextlib import ExitStack


class LayerScope:
    """ LayerScope

    Combines name_scope and var_scope and handles layer renaming if name already exists
    (since the name is used as a tensorflow name_scope)

    Args:
        layer: layer to be used in this scope, the layer name is used as scope name for tensorflow name_scope
        and variable_scope, also modifies the layer name if the scope name clashes with existing names

        reuse: if True does not change the input layer name but it does create a unique name for name_scope
        (for debug purposes only)
        var_scope: if True creates a variable_scope along with the named scope
        var_reuse: if True the variable scopes are created with the reuse option
        var_scope_name: if not None uses this name instead of the layer unique layer.scoped_name as a var scope
    """

    def __init__(self, layer, values=None, var_scope=None, var_reuse=False, var_scope_name=None, reuse=False):
        self.reuse = reuse
        self.var_scope_name = var_scope_name
        self.layer = layer
        self.values = values
        self.var_scope = var_scope
        self._stack = None
        self.var_reuse = var_reuse

    def __enter__(self):
        with ExitStack() as stack:

            # default_graph = ops.get_default_graph()
            # scoped_name = default_graph.unique_name(self.layer.name, False)
            # unscoped_name = scoped_name[scoped_name.find(self.layer.name):]

            # create new scope based on the layer unique name without scope
            # but take the current scope into account
            # this guarantees that reuse will not chain scoped names
            # like scope2/layer1/scope1/layer1 ...
            layer_name_scope = name_scope(self.layer.name, values=self.values)

            scoped_name = stack.enter_context(layer_name_scope)
            scoped_name = scoped_name[:-1]
            unique_unscoped_name = scoped_name[scoped_name.find(self.layer.name):]

            if not self.reuse:
                self.layer.name = unique_unscoped_name
                self.layer.scoped_name = scoped_name

            if self.var_scope:
                if self.var_scope_name is None:
                    self.var_scope_name = self.layer.scoped_name
                layer_var_scope = pure_variable_scope(self.var_scope_name, reuse=self.var_reuse)
                stack.enter_context(layer_var_scope)

            self._stack = stack.pop_all()

            return self.layer.scoped_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)


# alias for layer scopes
layer_scope = LayerScope


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
        scoped_name: layer name with its full scope (if created inside another scope)
        variable_names: a list of `tf.Variable` fully qualified name `layer_name/var_name'
        variables: a list of `tf.Variable` instances

    Note:
        Variables can be re-used elsewhere based on variable_names as follows::
            var_names = layer.var_names
            some_var = var_names[0]

            with tf.variable_scope('',reuse=True):
                v = tf.get_variable(some_var)


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
        self.scoped_name = name
        self.dtype = dtype
        self.input_layers = _as_list(input_layers)

        if shape is None:
            self.shape = [None, n_units]
        else:
            if shape[1] < n_units:
                raise ValueError("Shape mismatch: shape[1] < n_units")
            self.shape = shape

        # stores the variables if this layer has any
        self.variable_names = []
        self.variables = []

        self._tensor = None

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        """ Tensor Attribute Setter

        Prevents the setting of tensor to anything other than Tensor or Sparse Tensor
        """
        if not isinstance(tensor, (Tensor, SparseTensor)):
            raise TypeError(
                "tensor can only be set to Tensor or SparseTensor: {} found ".format(type(self.tensor)))
        self._tensor = tensor

    def _add_variable(self, var):
        if isinstance(var, variables.Variable):
            self.variables.append(var)
        self.variable_names.append(var.op.name)

    def is_sparse(self):
        """ Checks if the current layer is sparse

        A layer is sparse if its output tensor is a ``SparseTensor``, it is dense if the output tensor is a ``Tensor``.

        Returns:
            ``bool``: returns True if the current layer is sparse, False otherwise.

        """
        return isinstance(self.tensor, SparseTensor)

    def __str__(self):
        """ Informal string representation for a layer consists of Layer Class name, number of units and if its
        Sparse or Dense.

        Returns:
            :obj:`str`: a :obj:`str` with the informal representation for a layer instance.

        """
        class_name = type(self).__name__
        sparse_dense = "[Sparse]" if self.is_sparse() else "[Dense]"
        return "{layer_name}::{class_name}({n_units},{dtype}){sparse_dense}".format(class_name=class_name,
                                                                                    n_units=self.n_units,
                                                                                    dtype=self.dtype,
                                                                                    sparse_dense=sparse_dense,
                                                                                    layer_name=self.scoped_name)

    def full_str(self):
        """ Informal string representation for a layer that includes inner variables
        """
        fullstr = [str(self)]
        if len(self.variable_names) > 0:
            fullstr.append("variables:")
            for var_name in self.variable_names:
                fullstr.append("\t{var_name}".format(var_name=var_name))

        full_str = "\n".join(fullstr)
        return full_str + "\n"


class WrapLayer(Layer):
    """ Wraps another layer with tf code

    Utility layer used to wrap arbitrary layers with another tensorflow graph op
    this might be useful to customize existing layers without creating a new layer from scratch

    Attributes:
        tensor: like any other layer, the tensor is the application of the given tensorflow op to the output of the
        given layer
        placeholder: if the given layer is feedable (has the placeholder attribute) forwards that attribute (useful to
        create custom input pipelines

    Args:
        layer: a `Layer` to be wrapped by this Layer
        n_units: the new number of units this layer will have
        tf_fn: a callable returning a `Tensor` or `SparseTensor`
        name: name for this layer, defaults to wrap_[layer]


    """

    def __init__(self, layer, n_units, tf_fn, name="wrap"):
        if name == "wrap":
            name = "wrap_{}".format(layer.name)

        self.tf_fn = tf_fn

        if hasattr(layer, "placeholder"):
            self.placeholder = layer.placeholder

        self.variable_names = layer.variable_names
        self.variables = layer.variables

        shape = [layer.shape[0], n_units]
        super().__init__(layer, n_units, shape, dtype=layer.tensor.dtype, name=name)
        self.tensor = self._build_graph(layer)

    def _build_graph(self, layer):
        with layer_scope(self):
            tensor = self.tf_fn(layer.tensor)
            output_shape = tensor.get_shape()
            if not output_shape.is_compatible_with(self.shape):
                raise ValueError("shape from tf_fn {s1} incompatible with {s2}".format(s1=output_shape, s2=self.shape))

            if layer.dtype != tensor.dtype:
                self.dtype = tensor.dtype
        return tensor


class Compose(Layer):
    """ Compose Layer.

    Warning:
        all composed layers must be reusable, meaning that we should be able to call
        reuse_with in all the input layers.

    Composes two or more layers layer1.layer2 into a single layer reusable layer
    if the first layer is feedable, forwards the placeholder
    to the wrapper layer so that we know that the layer needs to be fed.
    All the vars are also forwarded to the compose layer so that we capture all
    the variables involved in the composition
    """

    def __init__(self, layers, name="compose"):

        self.layers = layers

        layer1, *layers = layers
        # if layers were not connected, connect them
        out_layer = layer1

        shape = [layer1.shape[0], out_layer.n_units]

        super().__init__(input_layers=layer1.input_layers,
                         n_units=out_layer.n_units,
                         shape=shape,
                         dtype=out_layer.dtype,
                         name=name)

        for curr_layer in layers:
            if out_layer not in curr_layer.input_layers:
                curr_layer = curr_layer.reuse_with(out_layer)
            out_layer = curr_layer

        # forward any feedable layer from the first layer
        if hasattr(layer1, "placeholder"):
            self.placeholder = layer1.placeholder

        # add both layer variables to the current layer container
        for var in itertools.chain.from_iterable([layer.variables for layer in self.layers]):
            self._add_variable(var)

        self.tensor = out_layer.tensor

    def reuse_with(self, input_layer, name=None):
        if name is None:
            name = self.name

        layer1, *layers = self.layers
        layer1 = layer1.reuse_with(input_layer)
        layers = [layer1] + layers
        return Compose(layers, name=name)

    def __str__(self):
        """ Informal string representation for a layer consists of Layer Class name, number of units and if its
        Sparse or Dense.

        Returns:
            :obj:`str`: a :obj:`str` with the informal representation for a layer instance.

        """
        class_name = type(self).__name__
        sparse_dense = "[Sparse]" if self.is_sparse() else "[Dense]"
        result = "{layer_name}::{class_name}({n_units},{dtype}){sparse_dense}".format(class_name=class_name,
                                                                                      n_units=self.n_units,
                                                                                      dtype=self.dtype,
                                                                                      sparse_dense=sparse_dense,
                                                                                      layer_name=self.scoped_name)

        inner_layers = [str(layer) for layer in self.layers]
        result = "{}\n \t{}".format(result, "\n\t".join(inner_layers))
        return result


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

        with layer_scope(self):
            # if n_active is not None convert to SparseTensor
            if n_active is None:
                self.placeholder = array_ops.placeholder(dtype=self.dtype, shape=self.shape, name=self.name)
                self.tensor = self.placeholder
            else:
                self.n_active = n_active
                self.placeholder = array_ops.placeholder(dtype=self.dtype, shape=[batch_size, n_active], name=self.name)
                self.tensor = transform.sparse_one_hot(self.placeholder, self.shape[1], dtype=self.dtype)

    def __str__(self):
        class_name = type(self).__name__
        if self.is_sparse():
            str_representation = "{class_name}({n_active}/{n_units},{dtype})[Sparse]".format(class_name=class_name,
                                                                                             n_active=self.n_active,
                                                                                             n_units=self.n_units,
                                                                                             dtype=self.dtype)
        else:
            str_representation = "{class_name}({n_units},{dtype})[Dense]".format(class_name=class_name,
                                                                                 n_units=self.n_units,
                                                                                 dtype=self.dtype)
        return str_representation


class TensorLayer(Layer):
    """ Tensor Input Layer

    Attributes:
        tensor: the tensor to be wrapped by this layer
        var_list: if vars are involved in the output tensor, they can be specified here
        and will be listed in variable_names and variables
        n_units: number of units for this layer,
        batch_size: Optional batch size for this layer

    Creates a layer from a given tensor that one can then integrate with other layers
    """

    def __init__(self, tensor, n_units, batch_size=None, var_list=None, dtype=dtypes.float32, name="tensor_input"):
        tensor = txutils.to_tensor_cast(tensor, dtype)

        if batch_size is not None:
            shape = [batch_size, n_units]
        else:
            shape = tensor.get_shape()
            shape.assert_is_compatible_with([batch_size, n_units])
            shape = shape.as_list()

            # if dynamic shape can't be determined, use the supplied values
            if all(dim is None for dim in shape):
                shape = [batch_size, n_units]

        if var_list is not None:
            for var in var_list:
                self._add_variable(var)

        super().__init__(None, n_units, shape, dtype, name)

        self.tensor = tensor


class SparseInput(Layer):
    """ Sparse Input Layer.

    Creates an op that depends on a `sparse_placeholder` to which one must feed a ``SparseTensorValue``.

    Attributes:
        placeholder: like all feedable layer, this has a placeholder attribute which one can (must) supply to feed_dict
        tensor: as with all layers there's a tensor attribute, not necessarily equivalent to the placeholder one

    Args:
        n_units (int): the number of output units for this layer
        batch_size (int): batch_size for the input, helps to define the shape for this sparse layer
        dtype: the output type for the values in the ``SparseTensor`` that this layer outputs
        name: name for the layer


    Notes:
        ``SparseTensorValues`` can be created with empty values, but this layer will require the number of values
        to be the same as the number of indices. If this is not the case, an ``InvalidArgumentError`` will be thrown
        when the `TensorFlow` graph is evaluated.
    """

    def __init__(self, n_units, batch_size=None, dtype=dtypes.float32, name="sparse_input"):
        shape = [batch_size, n_units]
        super().__init__(None, n_units, shape, dtype, name)

        with layer_scope(self):
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
                 shared_weights=None,
                 bias=True,
                 dtype=dtypes.float32,
                 name="linear", share_vars_with=None):

        self.shared_weights = shared_weights
        self.init = init
        self.bias = bias
        self.share_vars_with = share_vars_with

        shape = [layer.shape[1], n_units]
        super().__init__(layer, n_units, shape, dtype, name)

        if self.share_vars_with is not None:
            if not isinstance(self.share_vars_with, Linear):
                raise TypeError("Layer can only share variables with other layer of the same type")

            if self.shape != self.share_vars_with.shape:
                raise ValueError("Can only share variables with layers with the same shape: "
                                 "share_vars_with is provided but \n"
                                 "self shape: {s0} different from "
                                 "other shape: {s1}".format(s0=self.shape, s1=self.share_vars_with.shape))

        # if weights are passed, check that their shape matches the layer shape
        if self.shared_weights is not None:
            weights_shape = self.shared_weights.get_shape()
            if weights_shape[-1] != n_units:
                raise ValueError(
                    "weight shape mismatch: layer expects (,{}), provided weights have (,{})".format(n_units,
                                                                                                     weights_shape[-1]))

        self.tensor = self._build_graph(layer)

    def _build_graph(self, input_layer):
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with name_scope(name) as scope, variable_scope.variable_scope(scope[:-1]):
            # init weights

            if self.shared_weights is None:
                self.weights = variable_scope.get_variable("w",
                                                           shape=self.shape,
                                                           dtype=self.dtype,
                                                           initializer=self.init)
            else:
                self.weights = self.shared_weights

            # store variables for easy access
            self._add_variable(self.weights)
            # y = xW
            if input_layer.is_sparse():
                sp_values = input_layer.tensor
                sp_indices = transform.sparse_indices(sp_values)

                lookup_sum = embedding_lookup_sparse(params=self.weights,
                                                     sp_ids=sp_indices,
                                                     sp_weights=sp_values,
                                                     combiner="sum",
                                                     name=self.scoped_name + "_embeddings")
                tensor = lookup_sum
            else:
                tensor = math_ops.matmul(input_layer.tensor, self.weights, name="mat_mul")

            # y = xW + [b]
            if self.bias:
                self.bias = variable_scope.get_variable("b",
                                                        shape=[self.n_units],
                                                        dtype=self.dtype,
                                                        initializer=zero_init())
                self._add_variable(self.bias)
                tensor = bias_add(tensor, self.bias, name="add_b")
        return tensor

    def reuse_with(self, input_layer, name=None):
        """ Reuses the current layer on a different input.

        Uses the variables in this layer to create a new Layer instance with a different input_layer

        Args:
            input_layer: a ``Linear` layer
            name: name for the new ``Layer``

        Return:
            ``Layer``: a new layer with shared variables with the current layer.

        """
        # if current layer is sharing variables, forward the sharing
        share_vars_with = self.share_vars_with
        if share_vars_with is None:
            share_vars_with = self

        if name is None:
            name = self.name

        return Linear(layer=input_layer,
                      n_units=self.n_units,
                      init=self.init,
                      shared_weights=self.shared_weights,
                      bias=self.bias,
                      name=name,
                      share_vars_with=share_vars_with)


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
        A ``Tensor`` with shape [batch_size,seq_size*n_features] with the features of all elements in the sequence
        concatenated,

    Args:
        layer: an ``Input`` layer or ``SparseInput`` layers.
        seq_size: size of the sequence to be looked-up
        feature_shape: lookup table feature dimension
        batch_size: number of sequences to be looked up,
        if not None, will force a padding up to the specified batch_size

    """

    # TODO adaptive feature shape based on input if input has n_active
    def __init__(self,
                 layer,
                 seq_size,
                 feature_shape,
                 weight_init=random_uniform(),
                 batch_size=None,
                 dtype=dtypes.float32,
                 name="seq_lookup",
                 share_vars_with=None):

        self.weight_init = weight_init
        self.feature_shape = feature_shape
        self.seq_size = seq_size
        n_units = seq_size * feature_shape[-1]
        self.batch_size = batch_size
        shape = [batch_size, n_units]
        self.share_vars_with = share_vars_with

        super().__init__(layer, n_units, shape, dtype, name)

        if self.share_vars_with is not None:
            if not isinstance(self.share_vars_with, Lookup):
                raise TypeError("Layer can only share variables with other layer of the same type (Lookup)")

            if self.shape != self.share_vars_with.shape:
                raise ValueError("Can only share variables with layers with the same shape: "
                                 "share_vars_with is provided but \n"
                                 "self shape: {s0} different from "
                                 "other shape: {s1}".format(s0=self.shape, s1=self.share_vars_with.shape))

        # if weights are passed, check that their shape matches the layer shape

        self.tensor = self._build_graph(layer)

    def _build_graph(self, layer):
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with name_scope(name) as scope, variable_scope.variable_scope(scope):
            # init weights

            self.weights = variable_scope.get_variable("w", shape=self.feature_shape, initializer=self.weight_init)
            self._add_variable(self.weights)

            # batch size is dynamic and should be computed here because we need it
            # y = xW
            if layer.is_sparse():
                # dynamic batch size
                if self.batch_size is None:
                    batch_size = math_ops.ceil(array_ops.shape(layer.tensor)[0] / self.seq_size)
                else:
                    batch_size = self.batch_size

                sp_values = layer.tensor
                sp_indices = transform.sparse_indices(sp_values)

                # sums the lookups for the same row
                lookup_sum = embedding_lookup_sparse(params=self.weights,
                                                     sp_ids=sp_indices,
                                                     sp_weights=sp_values,
                                                     combiner="sum")

                flat_lookup = array_ops.reshape(lookup_sum, [-1])
                filled = array_ops.shape(flat_lookup)[0]

                # for sparse tensors this is int64
                batch_size = math_ops.cast(batch_size, dtypes.int32)

                fill_diff = (self.n_units * batch_size) - filled
                padding_shape = [math_ops.maximum(fill_diff, 0)]
                padding = array_ops.zeros(padding_shape)
                flat_lookup = array_ops.concat([flat_lookup, padding], axis=-1)

                # tensor = array_ops.reshape(lookup_sum, [-1, self.n_units])
                tensor = array_ops.reshape(flat_lookup, [-1, self.n_units])
            else:
                # if dense batch size is known
                if self.batch_size is None:
                    batch_size = array_ops.shape(layer.tensor)[0]
                else:
                    batch_size = self.batch_size

                lookup = embedding_lookup(params=self.weights,
                                          ids=layer.tensor)

                flat_lookup = array_ops.reshape(lookup, [-1])
                filled = array_ops.shape(flat_lookup)[0]

                padding_shape = [math_ops.maximum(self.n_units * batch_size - filled, 0)]
                padding = array_ops.zeros(padding_shape)
                flat_lookup = array_ops.concat([flat_lookup, padding], axis=-1)

                # tensor = array_ops.reshape(lookup, [self.batch_size, -1])
                tensor = array_ops.reshape(flat_lookup, [-1, self.n_units])

        return tensor

    def reuse_with(self, input_layer, name=None):
        """ Reuses the current layer on a different input.

        Uses the variables in this layer to create a new Layer instance with a different input_layer

        Args:
            input_layer: a ``Lookup` Layer
            name: name for the new ``Layer``

        Return:
            ``Layer``: a new layer with shared variables with the current layer.

        """
        # if current layer is sharing variables, forward the sharing
        share_vars_with = self.share_vars_with
        if share_vars_with is None:
            share_vars_with = self

        if name is None:
            name = self.name

        return Lookup(input_layer,
                      seq_size=self.seq_size,
                      feature_shape=self.feature_shape,
                      batch_size=self.batch_size,
                      weight_init=None,
                      dtype=self.dtype,
                      name=name,
                      share_vars_with=share_vars_with)


class Gate(Layer):
    """ Creates a Gate Layer that filters a given input layer using
    learned features from that layer.

    Warning:
        layer.n_units must be a multiple of n_units because if n_units < layer.n_units, we perform
        the gating by reshaping the input layer and using broadcasting.

        Example: if input layer has 6 units and gate has 2, it will apply the gating mechanism to a
        [-1,2,3] meaning the batch dimension will be the same but each gating unit will modulate 3
        input units at a time.

    Note:
        The original description (see reference) uses a re-scaled sigmoid (by 2) to make sure the gates
        output 1 when the input weights are 0, this is to guarantee that initially the network outputs
        the same as other models without gating (to allow for comparison with initial conditions), I make
        no such assumption with the default values of this layer -- although gate_fn can be specified to
        mimic this behaviour.

    Reference:
        (Mnih, et al. 2009) "Improving a statistical language model by modulating the effects of context words"

    Args:
            layer: a Layer to be gated
            n_units: number of gate units, the number of units of the layer to be gated should be a multiple of the
                    number of gate units (see Warning)
            h_dim: dimension for gate hidden unit
            gate_input: a layer to be used as the gate input
            h_fn: function for hidden layer
            gate_fn: function for gate
            shared_gate: if another gate is provided use the gate variables from that gate instead
    """

    def _apply_gate(self, layer, gate_tensor):
        feature_dim = layer.n_units // self.n_gates
        if layer.is_sparse():

            tensor_in = sparse_ops.sparse_reshape(layer.tensor, [-1, self.n_gates, feature_dim])
            gated = mathx.sparse_multiply_dense(tensor_in, array_ops.expand_dims(gate_tensor, -1))
        else:
            tensor_in = array_ops.reshape(layer.tensor, [-1, self.n_gates, feature_dim])
            gated = tensor_in * array_ops.expand_dims(gate_tensor, -1)

        return array_ops.reshape(gated, array_ops.shape(layer.tensor))

    def __init__(self, layer, n_gates, h_dim=None, gate_input=None, h_fn=elu, gate_fn=sigmoid, shared_gate=None,
                 name="gate"):
        super().__init__(layer, layer.n_units, layer.shape, dtype=dtypes.float32, name=name)

        self.h_dim = h_dim
        self.h_fn = h_fn
        self.gate_fn = gate_fn
        self.n_gates = n_gates
        self.gate_input = gate_input
        self.gate_weights = None
        self.gate_bias = None

        if h_dim is None and gate_input is None:
            raise ValueError("h_dim and gate_input cannot both be None")

        if gate_input is not None and h_dim is not None:
            if gate_input.n_units != h_dim:
                raise ValueError("Gate input n_units {} does not match h_dim {}".format(gate_input.n_units, h_dim))

        if shared_gate is not None and not isinstance(shared_gate, Gate):
            raise TypeError("shared_gate must be of type {} got {} instead".format(Gate, type(shared_gate)))

        self.shared_gate = shared_gate

        with layer_scope(self):
            if self.gate_input is None:
                h_l = Linear(layer, self.h_dim)
                h_a = Activation(h_l, h_fn)
                self.gate_input = Compose([h_l, h_a], name="gate_h")
            else:
                self.h_dim = self.gate_input.n_units

            if self.shared_gate is None:
                gate_l = Linear(self.gate_input, n_gates)
                gate_a = Activation(gate_l, gate_fn)
                self.gate = Compose([gate_l, gate_a], name="gate_units")

            else:
                self.gate = shared_gate.gate.reuse_with(self.gate_input, name="gate_units")

            self.gate_weights = self.gate.layers[-2].weights
            self.gate_bias = self.gate.layers[-2].bias
            tensor = self._apply_gate(layer, self.gate.tensor)

        self.tensor = tensor

        # add variables from all the inner gate layer
        for var in self.gate.variables:
            self._add_variable(var)

    def reuse_with(self, layer, gate_input=None, name=None):
        """ If we want to reuse this gate with a different
        gate_input, we must supply gate_input to reuse_with

        Args:
            layer: the new layer to be gated
            gate_input: the new gate_input

        Returns: ``Gate`` layer transforming the given layer
        with the current gate according to the current or supplied gate_input

        """
        if gate_input is None:
            gate_input = self.gate_input

        if name is None:
            name = self.name

        return Gate(layer=layer,
                    n_gates=self.n_gates,
                    h_dim=self.h_dim,
                    h_fn=self.h_fn,
                    gate_fn=self.gate_fn,
                    gate_input=gate_input,
                    shared_gate=self,
                    name=name)


class ToSparse(Layer):
    """ Transforms the previous layer into a sparse layer.

    This means that the current layer.tensor is a ``SparseTensor``
    """

    def __init__(self, layer):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_sparse")

        with layer_scope(self):
            if layer.is_sparse():
                tensor = layer.tensor
            else:
                tensor = transform.to_sparse(layer.tensor)

        self.tensor = tensor

    def reuse_with(self, layer):
        return ToSparse(layer)


class ToDense(Layer):
    """ ToDense transformation layer

    Transforms the previous layer into a dense layer (outputting a dense tensor)
    if the previous layer is already a dense layer, forwards the previous layer doing nothing

    """

    def __init__(self, layer):
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_dense")

        with layer_scope(self):
            if layer.is_sparse():
                tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                tensor = layer.tensor
        self.tensor = tensor

    def reuse_with(self, layer):
        return ToDense(layer)


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

    def __init__(self, layer, keep_prob=0.1, seed=None, name="dropout"):
        self.seed = seed
        self.keep_prob = keep_prob

        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, name)

        with layer_scope(self):
            if layer.is_sparse():
                tensor = transform.sparse_dropout(layer.tensor, self.keep_prob, seed)
            else:
                tensor = dropout(layer.tensor, self.keep_prob, seed=seed)

        self.tensor = tensor

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name
        return Dropout(layer, keep_prob=self.keep_prob, seed=self.seed, name=name)


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
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        super().__init__(input_layers=layer,
                         n_units=layer.n_units,
                         shape=layer.shape,
                         dtype=layer.dtype,
                         name=layer.name + "_gaussian_noise")

        with layer_scope(self):
            if layer.is_sparse():
                tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                tensor = layer.tensor

            noise_shape = array_ops.shape(tensor)
            noise = random_ops.random_normal(noise_shape, mean, stddev, seed=seed, dtype=dtypes.float32)

            tensor = math_ops.cast(tensor, dtypes.float32)
            tensor = math_ops.add(tensor, noise)

        self.tensor = tensor

    def reuse_with(self, layer):
        return GaussianNoise(layer, self.mean, self.stddev, self.seed)


class SparseGaussianNoise(Layer):
    """ Sparse Gaussian Noise Layer

    Applies additive gaussian noise to a given proportion of elements from an input layer.
    If ``density == 1.0``, this is equivalent to :class:`GaussianNoise`.

    """

    def __init__(self, layer, density=0.1, mean=0.0, stddev=0.2, dtype=None, seed=None):
        self.density = density
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

        self.dtype = layer.dtype
        if dtype is not None:
            self.dtype = dtype
        super().__init__(input_layers=layer,
                         n_units=layer.n_units,
                         shape=layer.shape,
                         dtype=self.dtype,
                         name=layer.name + "_sparse_gaussian_noise")

        with layer_scope(self):
            noise_shape = layer.shape
            noise = sparse_random_normal(noise_shape, density, mean, stddev, self.dtype, seed)

            if layer.is_sparse():
                tensor = transform.sparse_put(layer.tensor, noise)
            else:
                tensor = transform.dense_put(layer.tensor, noise)

        self.tensor = tensor

    def reuse_with(self, layer):
        return SparseGaussianNoise(layer, self.density, self.mean, self.stddev, self.dtype)


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
        self.density = density
        self.seed = seed
        self.salt_value = salt_value
        self.pepper_value = pepper_value

        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, layer.name + "_sp_noise")

        # do nothing if amount of noise is 0
        if density == 0.0 or self.num_corrupted() > 0:
            tensor = layer.tensor
        else:
            with layer_scope(self):
                noise_shape = layer.shape

                if noise_shape[0] is None:
                    batch_size = array_ops.shape(layer.tensor, out_type=dtypes.int64)[0]
                else:
                    batch_size = noise_shape[0]

                noise_shape = [batch_size, noise_shape[1]]
                noise = salt_pepper_noise(noise_shape, density, salt_value, pepper_value, seed)

                if layer.is_sparse(layer):
                    tensor = transform.sparse_put(layer.tensor, noise)
                else:
                    tensor = transform.dense_put(layer.tensor, noise)

        self.tensor = tensor

    def num_corrupted(self):
        """ Returns the number of entries corrupted by noise per sample"""
        noise_shape = self.shape
        num_noise = int(self.density * noise_shape[1])

        if num_noise < 2:
            num_noise = 0
        else:
            num_noise = (num_noise // 2) * 2
        return num_noise

    def reuse_with(self, layer):
        return SaltPepperNoise(layer,
                               self.density,
                               self.salt_value,
                               self.pepper_value,
                               self.seed)


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

    def __init__(self, layer, fn=array_ops.identity, name="activation", **keywords):
        self.fn = partial(fn, **keywords)
        self.kw = keywords
        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, name)

        with layer_scope(self):
            if layer.is_sparse():
                tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                tensor = layer.tensor
            tensor = self.fn(tensor)

        self.tensor = tensor

    def reuse_with(self, layer, name=None):
        name = self.name if name is None else name
        return Activation(layer, self.fn, name, **self.kw)


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

    def __init__(self, layer, name="bias", share_vars_with=None):

        self.share_vars_with = share_vars_with
        bias_name = "{}_{}".format(layer.name, name)

        super().__init__(layer, layer.n_units, layer.shape, layer.dtype, bias_name)

        if self.share_vars_with is not None:
            if not isinstance(self.share_vars_with, Bias):
                raise TypeError("Layer can only share variables with other layer of the same type")

            if self.n_units != self.share_vars_with.n_units:
                raise ValueError("Can only share variables with layers with the same n_units: "
                                 "share_vars_with is provided but \n"
                                 "self n_units: {s0} different from "
                                 "other n_units: {s1}".format(s0=self.n_units, s1=self.share_vars_with.n_units))

        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, values=[layer.tensor],
                         var_scope=True,
                         var_reuse=var_reuse,
                         var_scope_name=var_scope_name):
            self.bias = variable_scope.get_variable("b", shape=[self.n_units], initializer=zero_init())
            self._add_variable(self.bias)
            if layer.is_sparse():
                tensor = sparse_ops.sparse_tensor_to_dense(layer.tensor)
            else:
                tensor = layer.tensor
                tensor = bias_add(tensor, self.bias, name="tensor")

        self.tensor = tensor

    def reuse_with(self, layer, name=None):
        share_vars_with = self.share_vars_with if self.share_vars_with is not None else self

        if name is None:
            name = self.name

        return Bias(layer=layer,
                    name=name,
                    share_vars_with=share_vars_with)


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

    Example::
        out = tx.Merge([l1,l2],merge_fn=lambda tensors: tf.concat(tensors,axis=-1))

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

        self.weights = weights
        self.merge_fn = merge_fn

        if len(layers) < 2:
            raise Exception("Expecting a list of layers with len >= 2")

        if weights is not None and len(weights) != len(layers):
            raise Exception("len(weights) must be equals to len(layers)")

        super().__init__(layers, layers[0].n_units, layers[0].shape, layers[0].dtype, name)

        with layer_scope(self):
            if weights is not None:
                tensors = [math_ops.scalar_mul(weights[i], layers[i].tensor) for i in range(len(layers))]
            else:
                tensors = [layer.tensor for layer in layers]
            tensor = merge_fn(tensors)

        self.tensor = tensor


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

        with layer_scope(self):
            tensors = [layer.tensor for layer in layers]
            tensor = array_ops.concat(tensors, axis=-1)

        self.tensor = tensor


class SOMLinear(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 map_size,
                 map_distance=batch_cosine_distance,
                 init=random_uniform(), weights=None, bias=True, dtype=dtypes.float32, name="som_linear"):
        shape = [layer.shape[1], n_units]
        super().__init__(layer, n_units, shape, dtype, name)

        self.map_size = map_size
        self.weights_shape = [map_size, layer.shape[1], n_units]
        self.map_weights = None
        self.map_shape = [map_size, layer.shape[1]]
        self.bias = None

        with layer_scope(self):
            # init weights
            self.weights = variable_scope.get_variable("w", shape=self.weights_shape, initializer=init)
            self.map_weights = variable_scope.get_variable("som_w", shape=self.map_shape, initializer=init)

            if layer.is_sparse():
                self.tensor = None
            else:
                som_dist = batch_cosine_distance(layer.tensor, self.map_weights)
                # Best Matching Unit (BMU)
                bmu = math_ops.argmin(som_dist, axis=0)


__all__ = ["Input",
           "Gate",
           "Compose",
           "TensorLayer",
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
           "layers_to_list",
           "WrapLayer"
           ]
