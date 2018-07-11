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

from collections import deque

from tensorflow.python.framework import ops, dtypes
from tensorflow.python.framework.tensor_shape import TensorShape

from tensorflow.python.framework.ops import Tensor, name_scope
from tensorflow.python.ops import array_ops, check_ops

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.variable_scope import _pure_variable_scope as pure_variable_scope
from tensorflow.python.ops import variable_scope

from tensorflow.python.ops import random_ops, sparse_ops
from tensorflow.python.ops.nn import embedding_lookup_sparse, bias_add, dropout, embedding_lookup
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.init import random_uniform, zero_init, xavier_init
from tensorx.random import salt_pepper_noise, sparse_random_normal
from tensorx import transform
from tensorx import utils as txutils
from tensorx.metrics import batch_cosine_distance
from tensorx.activation import sigmoid, elu, tanh, identity
from tensorflow.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor

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

    def __init__(self, layer, name=None, values=None, var_scope=None, var_reuse=False, var_scope_name=None,
                 reuse=False):
        self.reuse = reuse
        self.var_scope_name = var_scope_name
        self.layer = layer
        if name is not None:
            self.layer.name = name
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


class Graph:
    """ Simple append only graph"""

    def __init__(self):
        self.nodes = set()
        self.edges_in = dict()
        self.edges_out = dict()

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges_in[node] = []
            self.edges_out[node] = []

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        self.edges_out[node1].append(node2)
        self.edges_in[node2].append(node1)


def _get_subgraph(input_layers, output_layers):
    input_layers = _as_list(input_layers)
    output_layers = _as_list(output_layers)

    endpoints = set()
    graph = Graph()

    def _update_graph(current_layer):
        in_layers = current_layer.input_layers

        if len(input_layers) > 0:
            if current_layer in input_layers:
                endpoints.add(current_layer)
                return True
        elif len(in_layers) == 0:
            endpoints.add(current_layer)
            return True

        path_found = {l: _update_graph(l) for l in in_layers}
        found = False
        terminals = set()
        for input_layer in path_found.keys():
            if path_found[input_layer]:
                graph.add_edge(input_layer, current_layer)
                found = found or True
            else:
                terminals.add(input_layer)

        # not all paths were valid, mark terminals
        if found and len(terminals) > 0:
            for terminal_layer in terminals:
                graph.add_edge(terminal_layer, current_layer)
                endpoints.add(terminal_layer)

        return found

    paths_found = [_update_graph(out_layer) for out_layer in output_layers]
    if not all(paths_found):
        failed = [output_layers[i] for i, path_found in enumerate(paths_found) if not path_found]
        failed_layers = "\n".join(failed)
        raise ValueError("no path found between inputs and layers: \n {}".format(failed_layers))

    return graph, endpoints


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


def layers_to_list(output_layers, input_layers=[]):
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
    output_layers = _as_list(output_layers)
    input_layers = _as_list(input_layers)

    graph, terminals = _get_subgraph(input_layers, output_layers)

    visited = set()
    flat_layers = list()
    stack = output_layers

    while stack:
        layer = stack.pop(0)
        if layer not in visited:
            visited.add(layer)
            flat_layers.append(layer)
            if layer in graph.edges_in:
                out_layers = graph.edges_in[layer]
                if len(out_layers) > 1:
                    pass
                    out_layers = out_layers[::-1]
                stack.extend(out_layers)

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
        # only sets the name if it hasn't been set before
        self.name = getattr(self, "name", name)
        self.scoped_name = name
        self.dtype = dtype
        self._input_layers = _as_list(input_layers)

        if shape is None:
            self.shape = [None, n_units]
        else:
            if shape[-1] != n_units:
                raise ValueError("Shape mismatch: shape[-1] != n_units")
            self.shape = shape

        # stores the variables if this layer has any
        self.variable_names = []
        self.variables = []

        self._tensor = None

    @property
    def input_layers(self):
        return list(self._input_layers)

    @input_layers.setter
    def input_layers(self, input_layers):
        raise ValueError("input_layers can't be set")

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
        return "{layer_name}::{class_name}({n_units},{dtype}){sparse_dense}".format(layer_name=self.scoped_name,
                                                                                    class_name=class_name,
                                                                                    n_units=self.n_units,
                                                                                    dtype=self.dtype,
                                                                                    sparse_dense=sparse_dense
                                                                                    )

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

    def __getitem__(self, item):
        return WrapLayer(self, self.n_units, tf_fn=lambda tensor: tensor[item])


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

    def __init__(self, layer, n_units, tf_fn, shape=None, name="wrap"):
        if name == "wrap":
            name = "wrap_{}".format(layer.name)

        self.tf_fn = tf_fn

        if hasattr(layer, "placeholder"):
            self.placeholder = layer.placeholder

        self.variable_names = layer.variable_names
        self.variables = layer.variables

        if shape is None:
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

    def reuse_with(self, layer):
        """ Reuse this WrapLayer with another layer

        # TODO this should be called ApplyLayer or FNLayer or something similar

        Reusing this layer is a matter of applying the tf graph building function
        to the new layer
        """
        return WrapLayer(layer, self.n_units, self.tf_fn)


class Module(Layer):
    """ Module Layer

    Defines a reusable module with multiple possible inputs and a single output

    Warnings:
        if any path from the layers in inputs does not lead to the output, this is an invalid module
        and an exception is raised when the Module is created

    Args:
        inputs: one or more input layers
        outputs: output layer
    """

    def __init__(self, inputs, output, name="module"):
        inputs = _as_list(inputs)

        graph, endpoints = _get_subgraph(inputs, output)
        self.graph = graph
        self.end_points = endpoints
        self.output = output

        super().__init__(input_layers=inputs,
                         n_units=output.n_units,
                         shape=output.shape,
                         dtype=output.dtype,
                         name=name)

        self.tensor = output.tensor

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name

        if len(layers) != len(self.input_layers):
            raise ValueError("Module has {} input layers, {} provided".format(len(self.input_layers)), len(layers))

        node_set = set()
        lstack = deque()

        def stack_out(in_layer):
            for out_layer in self.graph.edges_out[in_layer]:
                if out_layer not in node_set:
                    lstack.appendleft(out_layer)
                    node_set.add(out_layer)

        for in_layer in self.input_layers:
            stack_out(in_layer)

        # maps old layers to new layers
        layer_map = dict(zip(self.input_layers, layers))

        # transverse graph and apply reuse
        while len(lstack) > 0:
            current_layer = lstack.pop()

            # if layer exists in the map get it from the map
            new_inputs = [layer_map.get(in_layer, in_layer) for in_layer in self.graph.edges_in[current_layer]]
            new_layer = current_layer.reuse_with(*new_inputs)

            # map out how the current node corresponds to a new layer
            layer_map[current_layer] = new_layer
            # add descendants to the stack
            stack_out(current_layer)

        new_output = layer_map[self.output]

        return Module(inputs=layers, output=new_output, name=name)


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

    def __init__(self, *layers, name="compose"):

        self.layers = layers
        layer1, *layers = layers

        in_layer = layer1
        shape = in_layer.shape
        super().__init__(input_layers=layer1.input_layers,
                         n_units=in_layer.n_units,
                         shape=shape,
                         dtype=in_layer.dtype,
                         name=name)

        for curr_layer in layers:
            if in_layer not in curr_layer.input_layers:
                raise ValueError("\n Invalid Compose: \n {} --x {} \n [not connected]".format(in_layer, curr_layer))
                # curr_layer = curr_layer.reuse_with(out_layer)
            in_layer = curr_layer

        # forward any feedable layer from the first layer
        if hasattr(layer1, "placeholder"):
            self.placeholder = layer1.placeholder

        # add both layer variables to the current layer container
        for var in itertools.chain.from_iterable([layer.variables for layer in self.layers]):
            self._add_variable(var)

        self.tensor = in_layer.tensor

    def reuse_with(self, *input_layers, name=None):
        if name is None:
            name = self.name

        layer1, *layers = self.layers

        # check how many inputs for layer 1
        # this way we can compose with merge layers etc
        if len(layer1.input_layers) != len(input_layers):
            raise ValueError(
                "first layer of compose requires {} input layers {} passed".format(len(layer1.input_layers),
                                                                                   len(input_layers)))

        layer1 = layer1.reuse_with(*input_layers)

        new_layers = []
        # if layers were not connected, connect them
        in_layer = layer1
        for curr_layer in layers:
            if in_layer not in curr_layer.input_layers:
                curr_layer = curr_layer.reuse_with(in_layer)
                new_layers.append(curr_layer)
            in_layer = curr_layer

        return Compose(*new_layers, name=name)

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
            str_representation = "{layer_name}::{class_name}({n_active}/{n_units},{dtype})[Sparse]".format(
                layer_name=self.scoped_name,
                class_name=class_name,
                n_active=self.n_active,
                n_units=self.n_units,
                dtype=self.dtype)
        else:
            str_representation = "{layer_name}::{class_name}({n_units},{dtype})[Dense]".format(
                layer_name=self.scoped_name,
                class_name=class_name,
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

    def __init__(self, tensor, n_units, batch_size=None, var_list=None, dtype=None, name="tensor_input"):
        tensor = txutils.to_tensor_cast(tensor, dtype)
        dtype = tensor.dtype

        if batch_size is not None:
            shape = [batch_size, n_units]
        else:
            shape = tensor.get_shape()
            shape = shape.as_list()

            if shape[-1] != n_units or (batch_size is not None and shape[0] != batch_size):
                raise ValueError("tensor shape does not match expected batch size and n_units")

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
        shared_weights: None if we wish the layer to create a new variable or a tensorflow variable otherwise
        bias: if true creates a bias variable and the transformation becomes an affine transformation
        transpose_weights: if true transposes the weights of this linear layer. This is useful when we want to use tied
        weights but do not wish to transpose the weights passed to this layer, because transpose in tf is not a constant-time
        op.
        dtype: dtype for the output of this layer
        name: name for this layer scope
    """

    def __init__(self,
                 layer,
                 n_units,
                 init=random_uniform(),
                 shared_weights=None,
                 transpose_weights=False,
                 bias=True,
                 dtype=dtypes.float32,
                 name="linear", share_vars_with=None):

        self.shared_weights = shared_weights
        self.init = init
        self.bias = bias
        self.share_vars_with = share_vars_with
        self.transpose_weights = transpose_weights

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
            weight_shape = self.shared_weights.get_shape()
            if self.transpose_weights:
                if not TensorShape([layer.n_units]).is_compatible_with(TensorShape([weight_shape[-1]])):
                    raise ValueError(
                        "weight shape mismatch: input_layer shape {} :: weights shape {} with transpose_weights=True".format(
                            layer.shape,
                            weight_shape))
            else:
                if not TensorShape([layer.n_units]).is_compatible_with(TensorShape([weight_shape[0]])):
                    raise ValueError(
                        "weight shape mismatch: input_layer shape {} :: weights shape {} with transpose_weights=False".format(
                            layer.shape,
                            weight_shape))

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

                # if we use shared weights that must be transposed
                # but we have a sparse input to this layer, this is the most efficient way to do it
                if self.transpose_weights:
                    dense_sp = sparse_ops.sparse_tensor_to_dense(sp_values)
                    lookup_sum = math_ops.sparse_matmul(dense_sp, self.weights,
                                                        a_is_sparse=True,
                                                        transpose_b=True)
                else:

                    sp_indices = transform.sparse_indices(sp_values)
                    lookup_sum = embedding_lookup_sparse(params=self.weights,
                                                         sp_ids=sp_indices,
                                                         sp_weights=sp_values,
                                                         combiner="sum",
                                                         name=self.scoped_name + "_embeddings")
                tensor = lookup_sum
            else:
                tensor = math_ops.matmul(input_layer.tensor, self.weights, name="mat_mul",
                                         transpose_b=self.transpose_weights)

            # y = xW + [b]
            if self.bias:
                self.bias = variable_scope.get_variable("b",
                                                        shape=[self.n_units],
                                                        dtype=self.dtype,
                                                        initializer=zero_init())
                self._add_variable(self.bias)
                tensor = bias_add(tensor, self.bias, name="add_b")
        return tensor

    def reuse_with(self, input_layer, name=None, transpose_weights=None):
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

        if transpose_weights is None:
            transpose_weights = self.transpose_weights

        return Linear(layer=input_layer,
                      n_units=self.n_units,
                      init=self.init,
                      shared_weights=self.shared_weights,
                      transpose_weights=transpose_weights,
                      bias=self.bias,
                      name=name,
                      share_vars_with=share_vars_with)


class Fn(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 fn=identity,
                 init=random_uniform(),
                 shared_weights=None,
                 transpose_weights=False,
                 bias=True,
                 dtype=dtypes.float32,
                 name="fn",
                 share_vars_with=None):
        with layer_scope(self, name=name):
            self.linear = Linear(layer,
                                 n_units,
                                 init,
                                 shared_weights,
                                 transpose_weights,
                                 bias,
                                 dtype,
                                 "{}_linear".format(name),
                                 share_vars_with)

            self.activation = Activation(self.linear, fn=fn, name="{}_activation".format(name))

        super().__init__(input_layers=layer,
                         n_units=self.activation.n_units,
                         shape=self.activation.shape,
                         dtype=self.activation.dtype,
                         name=name)

        for var in self.linear.variables:
            self._add_variable(var)
        self.tensor = self.activation.tensor

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name

        return Fn(layer,
                  self.n_units,
                  self.activation.fn,
                  self.linear.init,
                  self.linear.shared_weights,
                  self.linear.transpose_weights,
                  self.linear.bias,
                  self.activation.dtype,
                  name,
                  self.linear.share_vars_with)


class RNNCell(Layer):
    """ Recurrent Cell
        Corresponds to a single step on an unrolled RNN network

        Args:
                layer: the input layer to the RNN Cell
                n_units: number of output units for this RNN Cell
                previous_state: a RNNCell from which we can extract output
                activation: activation function to be used in the cell
                use_bias: if True adds biases before the activation
                init: weight initialisation function
                recurrent_init: initialisation function for the recurrent weights
                share_state_with: a ``Layer`` with the same number of units than this Cell
                name: name for the RNN cell
        """

    def __init__(self, layer, n_units,
                 previous_state=None,
                 activation=tanh,
                 use_bias=True,
                 init=xavier_init(),
                 recurrent_init=xavier_init(),
                 share_state_with=None,
                 name="rnn_cell"):
        self.activation = activation
        self.use_bias = use_bias
        self.init = init

        self.recurrent_init = recurrent_init

        # if previous state is None start with zeros
        if previous_state is not None:
            if previous_state.n_units != n_units:
                raise ValueError(
                    "previous state n_units ({}) != current n_units ({})".format(previous_state.n_units, self.n_units))
        else:
            input_batch = array_ops.shape(layer.tensor)[0]
            zero_state = array_ops.zeros([input_batch, n_units])
            previous_state = TensorLayer(zero_state, n_units)

        self.previous_state = previous_state

        if share_state_with is not None and not isinstance(share_state_with, RNNCell):
            raise TypeError("shared_gate must be of type {} got {} instead".format(RNNCell, type(share_state_with)))
        self.share_state_with = share_state_with

        super().__init__([layer, previous_state], n_units, [layer.n_units, n_units], dtypes.float32, name)

        self.tensor = self._build_graph(layer, previous_state)

    def _build_graph(self, layer, previous_state):
        with layer_scope(self):

            if self.share_state_with is None:
                self.weights = Linear(layer, self.n_units, bias=True, init=self.init, name="w")
                self.recurrent_weights = Linear(previous_state, self.n_units, bias=False, init=self.recurrent_init,
                                                name="r_w")
            else:
                self.weights = self.share_state_with.weights.reuse_with(layer)
                self.recurrent_weights = self.share_state_with.recurrent_weights.reuse_with(previous_state)

            state = Add(self.weights, self.recurrent_weights)
            self.state = Activation(state, self.activation)

            return self.state.tensor

    def reuse_with(self, input_layer, previous_state=None, name=None):
        if previous_state is None:
            previous_state = self.previous_state

        if name is None:
            name = self.name

        return RNNCell(
            layer=input_layer,
            n_units=self.n_units,
            previous_state=previous_state,
            activation=self.activation,
            use_bias=self.use_bias,
            share_state_with=self,
            name=name
        )


class GRUCell(Layer):
    """ Gated Recurrent Unit Cell.

        Performs a single step with a gated recurrent unit where. These units have two gates:
        The first defines how much do we use the values from the recurrent connection to predict the current state
        The second
    """

    @staticmethod
    def zero_state(input_layer, n_units):
        input_batch = array_ops.shape(input_layer.tensor)[0]
        zero_state = array_ops.zeros([input_batch, n_units])
        return TensorLayer(zero_state, n_units)

    def __init__(self, layer, n_units,
                 previous_state=None,
                 activation=tanh,
                 use_bias=True,
                 init=xavier_init(),
                 recurrent_init=xavier_init(),
                 share_state_with=None,
                 name="rnn_cell"):
        self.activation = activation
        self.use_bias = use_bias
        self.init = init

        self.recurrent_init = recurrent_init

        # if previous state is None start with zeros
        if previous_state is not None:
            if previous_state.n_units != n_units:
                raise ValueError(
                    "previous state n_units ({}) != current n_units ({})".format(previous_state.n_units, self.n_units))
        else:
            previous_state = GRUCell.zero_state(layer, n_units)

        self.previous_state = previous_state

        if share_state_with is not None and not isinstance(share_state_with, GRUCell):
            raise TypeError("shared_gate must be of type {} got {} instead".format(GRUCell, type(share_state_with)))
        self.share_state_with = share_state_with

        super().__init__([layer, previous_state], n_units, [layer.n_units, n_units], dtypes.float32, name)

        self.tensor = self._build_graph(layer, previous_state)

    def _build_graph(self, layer, previous_state):
        with layer_scope(self):

            if self.share_state_with is None:
                # reset gate
                self.w_r = Linear(layer, self.n_units, bias=True, name="w_r")
                self.u_r = Linear(previous_state, self.n_units, bias=False, name="u_r")

                # candidate
                gate_r = Add(self.w_r, self.u_r, name="linear_r")
                gated_previous = Gate(previous_state, gate_r, name="gated_previous")

                self.w_h = Linear(layer, self.n_units, bias=True, init=self.init, name="w_h")
                self.u_h = Linear(gated_previous, self.n_units, bias=False, init=self.recurrent_init, name="u_h")

                linear_h = Add(self.w_h, self.u_h, name="linear_h")
                candidate_state = Activation(linear_h, self.activation, name="candidate")

                # coupled update gate
                self.w_z = Linear(layer, self.n_units, bias=True, name="w_z")
                self.u_z = Linear(previous_state, self.n_units, bias=False, name="u_z")
                update_gate = Add(self.w_z, self.u_z, name="linear_z")

                self.state = CoupledGate(candidate_state, previous_state, update_gate)
            else:
                # reset gate
                self.w_r = self.share_state_with.w_r.reuse_with(layer)
                self.u_r = self.share_state_with.u_r.reuse_with(previous_state)

                # candidate
                gate_r = Add(self.w_r, self.u_r)
                gated_previous = Gate(previous_state, gate_r)

                self.w_h = self.share_state_with.w_h.reuse_with(layer)
                self.u_h = self.share_state_with.u_h.reuse_with(gated_previous)

                candidate_state = Activation(Add(self.w_h, self.u_h), self.activation)

                # update gate
                self.w_z = self.share_state_with.w_z.reuse_with(layer)
                self.u_z = self.share_state_with.u_z.reuse_with(previous_state)

                update_gate = Add(self.w_z, self.u_z)

                self.state = CoupledGate(candidate_state, previous_state, update_gate)

            return self.state.tensor

    def reuse_with(self, input_layer, previous_state=None, name=None):
        if previous_state is None:
            previous_state = self.previous_state

        if name is None:
            name = self.name

        return GRUCell(
            layer=input_layer,
            n_units=self.n_units,
            previous_state=previous_state,
            activation=self.activation,
            use_bias=self.use_bias,
            share_state_with=self,
            name=name
        )


class LSTMCell(Layer):
    """ LSTM Cell

        Performs a single step with a gated recurrent unit where. These units have two gates:
        The first defines how much do we use the values from the recurrent connection to predict the current state
        The second
    """

    @staticmethod
    def zero_state(input_layer, n_units):
        input_batch = array_ops.shape(input_layer.tensor)[0]
        zero_state = array_ops.zeros([input_batch, n_units])
        return TensorLayer(zero_state, n_units)

    def __init__(self, layer, n_units,
                 previous_state=None,
                 memory_state=None,
                 candidate_activation=tanh,
                 output_activation=tanh,
                 init=xavier_init(),
                 recurrent_init=xavier_init(),
                 share_state_with=None,
                 name="lstm_cell"):

        self.candidate_activation = candidate_activation
        self.output_activation = output_activation
        self.init = init

        self.recurrent_init = recurrent_init

        # if previous state is None start with zeros
        if previous_state is not None:
            if previous_state.n_units != n_units:
                raise ValueError(
                    "previous state n_units ({}) != current n_units ({})".format(previous_state.n_units, self.n_units))
        else:
            previous_state = LSTMCell.zero_state(layer, n_units)

        if memory_state is not None:
            if memory_state.n_units != n_units:
                raise ValueError(
                    "previous memory_state n_units ({}) != current n_units ({})".format(memory_state.n_units,
                                                                                        self.n_units))
        else:
            memory_state = LSTMCell.zero_state(layer, n_units)

        self.previous_state = previous_state
        self.memory_state = memory_state

        if share_state_with is not None and not isinstance(share_state_with, LSTMCell):
            raise TypeError("shared_gate must be of type {} got {} instead".format(RNNCell, type(share_state_with)))
        self.share_state_with = share_state_with

        super().__init__([layer, previous_state, memory_state], n_units, [layer.n_units, n_units], dtypes.float32, name)

        self.tensor = self._build_graph(layer, previous_state)

    def _build_graph(self, layer, previous_state):
        with layer_scope(self):

            if self.share_state_with is None:
                # forget gate linear
                self.w_f = Linear(layer, self.n_units, bias=True)
                self.u_f = Linear(previous_state, self.n_units, bias=False)

                # input gate linear
                self.w_i = Linear(layer, self.n_units, bias=True)
                self.u_i = Linear(previous_state, self.n_units, bias=False)

                # candidate linear
                self.w_c = Linear(layer, self.n_units, bias=True)
                self.u_c = Linear(previous_state, self.n_units, bias=False)

                # output gate
                self.w_o = Linear(layer, self.n_units, bias=True)
                self.u_o = Linear(previous_state, self.n_units, bias=False)

            else:
                # forget gate linear
                self.w_f = self.share_state_with.w_f.reuse_with(layer)
                self.u_f = self.share_state_with.u_f.reuse_with(previous_state)

                # input gate linear
                self.w_i = self.share_state_with.w_i.reuse_with(layer)
                self.u_i = self.share_state_with.u_i.reuse_with(previous_state)

                # candidate linear
                self.w_c = self.share_state_with.w_c.reuse_with(layer)
                self.u_c = self.share_state_with.u_c.reuse_with(previous_state)

                # output gate
                self.w_o = self.share_state_with.w_o.reuse_with(layer)
                self.u_o = self.share_state_with.u_o.reuse_with(previous_state)

            # build gates
            gate_f = Add(self.w_f, self.u_f)
            gate_i = Add(self.w_i, self.u_i)
            gate_o = Add(self.w_o, self.u_o)

            memory_state = Gate(self.memory_state, gate_f)

            candidate = Activation(Add(self.w_c, self.u_c), fn=self.candidate_activation)
            candidate = Gate(candidate, gate_i)

            self.memory_state = Add(memory_state, candidate)

            output = Activation(memory_state, fn=self.output_activation)
            self.state = Gate(output, gate_o)

            return self.state.tensor

    def reuse_with(self, input_layer, previous_state=None, memory_state=None, name=None):
        if previous_state is None:
            previous_state = self.previous_state

        if memory_state is None:
            memory_state = self.memory_state

        if name is None:
            name = self.name

        return LSTMCell(
            layer=input_layer,
            n_units=self.n_units,
            previous_state=previous_state,
            memory_state=memory_state,
            candidate_activation=self.candidate_activation,
            output_activation=self.output_activation,
            name=name
        )


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
        as_sequence: if True returns a [seq_size, batch_size, feature_shape[-1]] tensor
    """

    # TODO adaptive feature shape based on input if input has n_active
    def __init__(self,
                 layer,
                 seq_size,
                 feature_shape,
                 weight_init=random_uniform(),
                 batch_size=None,
                 bias=False,
                 shared_bias=None,
                 shared_weights=None,
                 dtype=dtypes.float32,
                 name="seq_lookup",
                 share_vars_with=None,
                 as_sequence=False):

        self.weight_init = weight_init
        self.as_sequence = as_sequence
        self.feature_shape = feature_shape
        self.seq_size = seq_size

        self.bias = bias
        self.shared_bias = shared_bias

        n_units = feature_shape[-1]
        if not as_sequence:
            n_units = seq_size * n_units

        self.batch_size = batch_size
        shape = [batch_size, n_units]
        self.share_vars_with = share_vars_with
        self.shared_weights = shared_weights

        super().__init__(layer, n_units, shape, dtype, name)

        if self.shared_weights is not None:
            weight_shape = shared_weights.get_shape().as_list()
            if feature_shape != weight_shape:
                raise ValueError(
                    "shared weight shape {} and feature shape {} mismatch".format(weight_shape, feature_shape))

        if self.shared_bias is not None:
            num_bias = shared_bias.get_shape().as_list()[-1]
            if feature_shape[0] != num_bias:
                raise ValueError(
                    "number of bias {} and number of feature rows {} mismatch".format(num_bias, feature_shape[0]))

        if self.share_vars_with is not None:
            if not isinstance(self.share_vars_with, Lookup):
                raise TypeError("Layer can only share variables with other layer of the same type (Lookup)")

            if self.feature_shape != self.share_vars_with.feature_shape:
                raise ValueError("Can only share variables with layers with the same feature shape: "
                                 "share_vars_with is provided but \n"
                                 "self shape: {s0} different from "
                                 "other shape: {s1}".format(s0=self.feature_shape,
                                                            s1=self.share_vars_with.feature_shape))

        # if weights are passed, check that their shape matches the layer shape

        self.tensor = self._build_graph(layer)

    def _build_graph(self, layer):
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with name_scope(name) as scope, variable_scope.variable_scope(scope):
            # init weights

            if self.shared_weights is None:
                self.weights = variable_scope.get_variable("w", shape=self.feature_shape, initializer=self.weight_init)
            else:
                self.weights = self.shared_weights

            self._add_variable(self.weights)

            if self.bias:
                if self.shared_bias is None:
                    self.bias = variable_scope.get_variable("b", shape=self.feature_shape[0], initializer=zero_init())
                else:
                    self.bias = self.bias
            else:
                self.bias = None

            if self.bias is not None:
                self._add_variable(self.bias)

            # total number of features that should be looked up
            # used to compute padding if necessary
            n_features = self.n_units
            if self.as_sequence:
                n_features = self.n_units * self.seq_size

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
                lookup_weights = embedding_lookup_sparse(params=self.weights,
                                                         sp_ids=sp_indices,
                                                         sp_weights=sp_values,
                                                         combiner="sum")

                if self.bias is not None:
                    # lookup bias
                    lookup_bias = embedding_lookup_sparse(params=self.bias,
                                                          sp_ids=sp_indices,
                                                          sp_weights=sp_values,
                                                          combiner="sum")

                    lookup_bias = array_ops.expand_dims(lookup_bias, -1)

                    lookup_weights += lookup_bias

                flat_lookup = array_ops.reshape(lookup_weights, [-1])
                filled = array_ops.shape(flat_lookup)[0]

                # for sparse tensors this is int64
                batch_size = math_ops.cast(batch_size, dtypes.int32)

                fill_diff = (n_features * batch_size) - filled
                padding_shape = [math_ops.maximum(fill_diff, 0)]
                padding = array_ops.zeros(padding_shape)
                flat_lookup = array_ops.concat([flat_lookup, padding], axis=-1)

            else:
                # if dense batch size is known
                if self.batch_size is None:
                    batch_size = array_ops.shape(layer.tensor)[0]
                else:
                    batch_size = self.batch_size

                lookup_weights = embedding_lookup(params=self.weights,
                                                  ids=layer.tensor)

                if self.bias is not None:
                    lookup_bias = embedding_lookup(params=self.bias,
                                                   ids=layer.tensor)

                    lookup_bias = array_ops.expand_dims(lookup_bias, -1)
                    lookup_weights += lookup_bias

                flat_lookup = array_ops.reshape(lookup_weights, [-1])
                filled = array_ops.shape(flat_lookup)[0]

                padding_shape = [math_ops.maximum(n_features * batch_size - filled, 0)]
                padding = array_ops.zeros(padding_shape, dtype=self.weights.dtype)
                flat_lookup = array_ops.concat([flat_lookup, padding], axis=-1)

            tensor = array_ops.reshape(flat_lookup, [-1, n_features])
            if self.as_sequence:
                seqs = array_ops.split(tensor, self.seq_size, -1)
                tensor = array_ops.concat(seqs, 0)
                tensor = array_ops.reshape(tensor, [self.seq_size, -1, self.n_units])

        return tensor

    def reuse_with(self, input_layer, as_sequence=None, name=None):
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
        if as_sequence is None:
            as_sequence = self.as_sequence

        return Lookup(input_layer,
                      seq_size=self.seq_size,
                      feature_shape=self.feature_shape,
                      batch_size=self.batch_size,
                      shared_weights=self.shared_weights,
                      weight_init=None,
                      dtype=self.dtype,
                      name=name,
                      share_vars_with=share_vars_with,
                      as_sequence=as_sequence)


def _apply_gate(layer: Layer, gate: Layer):
    with ops.name_scope("apply_gate", values=[layer.tensor, gate.tensor]):
        n_gates = gate.n_units
        feature_dim = layer.n_units // n_gates
        if layer.is_sparse():

            tensor_in = sparse_ops.sparse_reshape(layer.tensor, [-1, n_gates, feature_dim])
            gated = mathx.sparse_multiply_dense(tensor_in, array_ops.expand_dims(gate.tensor, -1))
        else:
            tensor_in = array_ops.reshape(layer.tensor, [-1, n_gates, feature_dim])
            gated = tensor_in * array_ops.expand_dims(gate.tensor, -1)

        return array_ops.reshape(gated, array_ops.shape(layer.tensor))


class Gate(Layer):
    """ Creates a Gate Layer that filters a given input layer using
    learned features from that layer.

    Warning:
        layer.n_units must be a multiple of n_units because if n_units < layer.n_units, we perform
        the gating by reshaping the input layer and using broadcasting.

        Example: if input layer has 6 units and gate has 2, it will apply the gating mechanism to a
        [-1,2,3] meaning the batch dimension will be the same but each gating unit will modulate 3
        input units at a time.


    Args:
            layer: a Layer to be gated
            gate_input: a layer to be used as the gate input
            gate_fn: function for gate
    """

    def __init__(self, layer, gate_input, gate_fn=sigmoid, name="gate"):
        if layer.n_units % gate_input.n_units != 0:
            raise ValueError("the n_units of the input layer {} is not a multiple of gate n_units {}".format(
                layer.n_units, gate_input.n_units))

        super().__init__([layer, gate_input], layer.n_units, layer.shape, dtype=dtypes.float32, name=name)

        self.gate_fn = gate_fn
        self.gate_input = gate_input
        with layer_scope(self):
            self.gate = Activation(self.gate_input, self.gate_fn)
            self.tensor = _apply_gate(layer, self.gate)

    def reuse_with(self, layer, gate_input=None, name=None):
        if gate_input is None:
            gate_input = self.gate_input

        if name is None:
            name = self.name

        return Gate(layer=layer,
                    gate_input=gate_input,
                    gate_fn=self.gate_fn,
                    name=name)


class CoupledGate(Layer):
    """ Creates a Gate Layer that modulates between two layers using a gate:

    output = (r) * layer1 + (1-r) * layer2 where (r) is the gate output [0,1]

    Warning:
        layer.n_units must be a multiple of n_units because if n_units < layer.n_units, we perform
        the gating by reshaping the input layer and using broadcasting.

        Example: if input layer has 6 units and gate has 2, it will apply the gating mechanism to a
        [-1,2,3] meaning the batch dimension will be the same but each gating unit will modulate 3
        input units at a time.


    Args:
            layer1: first Layer
            layer2: second Layer
            gate_input: a layer to be used as the gate input (usually the linear part)
            gate_fn: function for gate
    """

    def __init__(self, layer1, layer2, gate_input, gate_fn=sigmoid, name="coupled_gate"):

        if not layer1.tensor.get_shape().is_compatible_with(layer2.tensor.get_shape()):
            raise ValueError("layers must have the same shape: {}!={}".format(layer1.shape, layer2.shape))

        if layer1.n_units % gate_input.n_units != 0:
            raise ValueError("the n_units of the input layer {} is not a multiple of gate n_units {}".format(
                layer1.n_units, gate_input.n_units))

        super().__init__([layer1, layer2, gate_input], layer1.n_units, layer1.shape, dtype=dtypes.float32, name=name)

        self.gate_fn = gate_fn
        self.gate_input = gate_input

        with layer_scope(self):
            self.gate1 = Activation(self.gate_input, self.gate_fn)
            self.gate2 = WrapLayer(self.gate1, self.gate1.n_units, lambda x: 1 - x)

            output1 = _apply_gate(layer1, self.gate1)
            output2 = _apply_gate(layer2, self.gate2)
            output = math_ops.add(output1, output2)
            self.tensor = output

    def reuse_with(self, layer1, layer2, gate_input=None, name=None):
        if gate_input is None:
            gate_input = self.gate_input

        if name is None:
            name = self.name

        return CoupledGate(layer1=layer1,
                           layer2=layer2,
                           gate_input=gate_input,
                           gate_fn=self.gate_fn,
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
                 *layers,
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

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name
        return Merge(*layers, weights=self.weights, merge_fn=self.merge_fn, name=name)


class Add(Merge):
    """ Adds the outputs of multiple layers with the same shape

    Args:
            layers: a list of layers with the same number of units to be merged
            weights: a list of weights
            name: name for layer scope
    """

    def __init__(self, *layers, weights=None, name="add"):
        super().__init__(*layers, weights=weights, merge_fn=math_ops.add_n, name=name)


class Concat(Layer):
    """ Concat Layer

    Concatenates input layers on the last dimension

    Args:
        layers: a :obj:`list` of :class:`Layer`
        name: name for the layer scope
    """

    def __init__(self, *layers, name="concat"):
        first, *rest = layers
        if not all(layer.dtype == first.dtype for layer in rest):
            raise ValueError("Layers must have the same type to be concatenated")

        total_units = sum([layer.n_units for layer in layers])
        super().__init__(layers, total_units, dtype=first.dtype, name=name)

        with layer_scope(self):
            tensors = [layer.tensor for layer in layers]
            tensor = array_ops.concat(tensors, axis=-1)

        self.tensor = tensor

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name
        return Concat(*layers, name)


__all__ = ["Input",
           "Fn",
           "RNNCell",
           "GRUCell",
           "LSTMCell",
           "Gate",
           "CoupledGate",
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
           "WrapLayer",
           "Module"
           ]
