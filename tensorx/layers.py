""" Neural Network Layers"""

import itertools
from collections import deque

import operator

from contextlib import ExitStack
from functools import partial, reduce
from typing import Callable, Union, Optional, List
import inspect
from tensorx import math as mathx
from tensorx import transform
from tensorx import utils as tx_utils
from tensorx.activation import sigmoid, tanh, identity
from tensorx.init import random_uniform, zeros_init, glorot_uniform, const_init, ones_init
from tensorx.math import embedding_lookup_sparse
from tensorx.random import salt_pepper_noise, sparse_random_normal, random_bernoulli
from tensorx.utils import Graph
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops.variable_scope import _pure_variable_scope as pure_var_scope
from tensorx.utils import as_list
from tensorx.callbacks import Property


def _validate_shape_type(x, shape, dtype=None):
    if x is not None:
        x = tx_utils.to_tensor_cast(x)
        tensor_shape = x.get_shape().as_list()
        if tensor_shape != shape:
            raise ValueError(
                "Invalid shape for {name} {tensor_shape}. Expected shape {expected}".format(name=str(x.name),
                                                                                            tensor_shape=tensor_shape,
                                                                                            expected=tensor_shape))
        if dtype is not None and x.dtype != dtype:
            raise TypeError("Invalid dtype for {name}. Expected {expected}, was {got} instead".format(name=str(x.name),
                                                                                                      expected=dtype,
                                                                                                      got=x.dtype))


class LayerScope:
    """ LayerScope

    Combines tf.name_scope and var_scope and handles layer renaming if name already exists
    (since the name is used as a tensorflow tf.name_scope)

    Args:
        layer: layer to be used in this scope, the layer name is used as scope name for tensorflow tf.name_scope
        and variable_scope, also modifies the layer name if the scope name clashes with existing names

        reuse: if True does not change the input layer name but it does create a unique name for tf.name_scope
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

            # default_graph = tf.get_default_graph()
            # scoped_name = default_graph.unique_name(self.layer.name, False)
            # unscoped_name = scoped_name[scoped_name.find(self.layer.name):]

            # create new scope based on the layer unique name without scope
            # but take the current scope into account
            # this guarantees that reuse will not chain scoped names
            # like scope2/layer1/scope1/layer1 ...

            layer_name_scope = tf.name_scope(self.layer.name)

            scoped_name = stack.enter_context(layer_name_scope)
            scoped_name = scoped_name[:-1]
            unique_unscoped_name = scoped_name[scoped_name.find(self.layer.name):]

            if not self.reuse:
                self.layer.name = unique_unscoped_name
                self.layer.scoped_name = scoped_name

            if self.var_scope:
                if self.var_scope_name is None:
                    self.var_scope_name = self.layer.scoped_name
                layer_var_scope = pure_var_scope(self.var_scope_name, reuse=self.var_reuse)
                stack.enter_context(layer_var_scope)

            self._stack = stack.pop_all()

            return self.layer.scoped_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)


# alias for layer scopes
layer_scope = LayerScope


def _get_subgraph(input_layers, output_layers):
    input_layers = as_list(input_layers)
    output_layers = as_list(output_layers)

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
        failed = [str(output_layers[i]) for i, path_found in enumerate(paths_found) if not path_found]
        failed_layers = "\n".join(failed)
        raise ValueError("no path found between input and output layers: \n {}".format(failed_layers))

    return graph, endpoints


class LayerProto:
    def _validate_args(self, **kwargs):
        for key in kwargs:
            if key not in self.args:
                raise TypeError("{} prototype got an unexpected argument {}".format(self.layer_cls.__name__, key))

    """ Layer Proto

    Utility class for the creation of Layer prototypes. A LayerProto is a callable that validates invalid layer
    constructor arguments.

    """

    def __init__(self, layer_cls, **kwargs):
        self.layer_cls = layer_cls

        spec = inspect.getfullargspec(layer_cls.__init__)
        self.args = spec.args[1:]
        self._validate_args(**kwargs)

        self.args_set = kwargs

    def __call__(self, *args, **kwargs):
        new_args = dict(self.args_set)
        new_args.update(kwargs)
        return self.layer_cls(*args, **new_args)

    def update(self, **kwargs):
        self._validate_args(**kwargs)
        self.args_set.update(kwargs)


class Layer:
    """ Layer.

    Features:
        Converts to input_layers to layers when these are tensors or anything convertible to tensor.

    Attributes:
        input_layers: a list of Layers that serve as input to the current layer
        n_units: the number of units for the current layer
        _tensor: a ``Tensor`` or ``SparseTensor`` if the layer is dense or sparse respectively
        dtype: the dtype for the output tensor
        name: a name used to build a named_scope for the layer
        scoped_name: layer name with its full scope (if created inside another scope)
        variables: a list of `tf.Variable` instances


    Args:
        input_layers: a single layer,a list of input layers, or None if no inputs are required
        n_units: dimension of input vector (dimension of columns in case batch_size != None
        dtype: expected input TensorFlow data type
        name: layer name (used to nam the placeholder)

    """

    def __init__(self, input_layers, n_units, dtype=tf.float32, name="layer"):
        self.n_units = n_units
        self.name = getattr(self, "name", name)
        self.scoped_name = name
        self.dtype = dtype
        self._input_layers = [convert_to_layer(layer) for layer in as_list(input_layers)]

        # stores the variables if this layer has any
        self.variables = []

        self.attr_names = []

        self._tensor = None

    @classmethod
    def proto(cls, **kwargs):
        return LayerProto(cls, **kwargs)

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name
        return type(self)(*layers, name=name)

    @property
    def shape(self) -> List[Union[None, int]]:
        """

        Returns:
            shape (list): list representing the static shape with int dimensions or None if a dimension is unknown.

        """
        static_shape = self.tensor.get_shape()

        return static_shape.as_list()

    @shape.setter
    def shape(self, shape):
        raise ValueError("a static shape cannot be set")

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
        if not isinstance(tensor, (tf.Tensor, tf.SparseTensor, tf.Variable, tf.Operation)):
            raise TypeError(
                "tensor can only be set to Tensor or tf.SparseTensor: {} found ".format(type(self.tensor)))
        self._tensor = tensor

    def _add_variable(self, var):
        if isinstance(var, tf.Variable):
            self.variables.append(var)

    def is_sparse(self):
        """ Checks if the current layer is sparse

        A layer is sparse if its output tensor is a ``SparseTensor``, it is dense if the output tensor is a ``Tensor``.

        Returns:
            ``bool``: returns True if the current layer is sparse, False otherwise.

        """
        return isinstance(self.tensor, tf.SparseTensor)

    def is_dense(self):
        """ Checks if the current layer is dense
        """
        return not self.is_sparse()

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
        full_str = [str(self)]
        if len(self.variables) > 0:
            full_str.append("variables:")
            for var in self.variables:
                full_str.append("\t{var_name}".format(var_name=var.name))

        full_str = "\n".join(full_str)
        return full_str + "\n"

    def __getitem__(self, item):
        if isinstance(item, tf.Tensor):
            item_name = item.op.name
        else:
            item_name = str(item)
        return WrapLayer(layer=self,
                         n_units=self.n_units,
                         wrap_fn=lambda tensor: tensor[item],
                         name="{}_item_{}".format(self.name, item_name))

    def eval(self, feed_dict=None, session=None):
        if isinstance(self.tensor, tf.Variable):
            return self.tensor.eval(session=session)
        else:
            return self.tensor.eval(feed_dict=feed_dict, session=session)

    def __call__(self, *args, **kwargs):
        return type(self).reuse_with(self, *args, **kwargs)


class WrapLayer(Layer):
    """ Wraps another layer with tf code

    Utility layer used to wrap arbitrary layers with another tensorflow graph op
    this might be useful to customize existing layers without creating a new layer from scratch


    Example::

    You can create nested WrapLayer objects in which case, the reuse will forwarded all the way to the input layer

                     +---------------------------------------+
                     | +----------------------------+        |
                     | | +------------+             |        |
                     | | |            | WRAP        | WRAP   |
                     | | |   INPUT    |             |        |
            +--------------> LAYER    |             |        +------->
                     | | |            |             |        |
                     | | +------------+             |        |
                     | +----------------------------+        |
                     +---------------------------------------+


    Attributes:
        tensor: like any other layer, the tensor is the application of the given tensorflow op to the output of the
        given layer
        placeholder: if the given layer is feedable (has the placeholder attribute) forwards that attribute (useful to
        create custom input pipelines

    Args:
        layer: a `Layer` to be wrapped by this Layer
        n_units: the new number of units this layer will have
        wrap_fn: a callable returning a `Tensor` or `SparseTensor`
        name: name for this layer, defaults to wrap_[layer]
        layer_fn: if False applies the function to the layer tensor outputs
        if false applies to the layer itself. if False applies to the layer itself and expects
        the output to be a tensor.



    """

    def __init__(self, layer, wrap_fn, n_units=None, attr_fwd=None, name="wrap", layer_fn=False):
        self.layer_fn = layer_fn
        if name == "wrap":
            name = "wrap_{}".format(layer.name)

        self.wrap_fn = wrap_fn

        if hasattr(layer, "placeholder"):
            self.placeholder = layer.placeholder

        self.attr_fwd = as_list(attr_fwd)
        for attr in self.attr_fwd:
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

        self.variables = layer.variables

        with layer_scope(self, name=name):
            fn_inputs = layer.tensor if not layer_fn else layer
            tensor = self.wrap_fn(fn_inputs)
            dtype = tensor.dtype if not isinstance(tensor, tf.Operation) else None
            fn_n_units = tensor.get_shape().as_list()[-1]

            if n_units is not None and fn_n_units != n_units:
                ValueError("provided n_units and result wrap_fn resulting tensor last dimension do not match")
            if n_units is None:
                n_units = fn_n_units

        super().__init__(input_layers=layer, n_units=n_units, dtype=dtype, name=name)

        self.tensor = tensor

    def reuse_with(self, *layers, name=None):
        """ Reuse with a different input layer

            Calls reuse with on the wrapped layer and then creates a new wrapped layer
            around it, using the current tensor function.
        """
        new_wrapped = self.input_layers[0].reuse_with(*layers)

        # forward any previous attributes if we're wrapping over other WrapLayer instances
        attr_fwd = self.attr_fwd
        if isinstance(new_wrapped, WrapLayer):
            attr_fwd += new_wrapped.attr_fwd

        if name is None:
            name = self.name

        return WrapLayer(layer=new_wrapped,
                         n_units=self.n_units,
                         wrap_fn=self.wrap_fn,
                         attr_fwd=attr_fwd,
                         name=name,
                         layer_fn=self.layer_fn)


class VariableLayer(Layer):
    """
    Warning:
        dynamic shape inference is broken by this layer if we use a resource variable. If we use a normal variable,
        we can set the shape so that the graph knows that henceforth, the shape of the first dimension is undetermined.

        If used with another layer (e.g. Linear) as a set of shared weights, this layer will have an unknown number of
        output units.
    """

    def __init__(self,
                 input_layer=None,
                 n_units=None,
                 var_shape=None,
                 init_from_input=False,
                 trainable=False,
                 resource=False,
                 dtype=tf.float32,
                 init=None,
                 share_vars_with=None,
                 name="variable"):
        self.share_vars_with = share_vars_with
        self.init_from_input = init_from_input

        if input_layer is not None:
            if not isinstance(input_layer, Layer):
                input_layer = TensorLayer(input_layer)
            if n_units is not None and n_units != input_layer.n_units:
                raise ValueError("n_units must match input_layer.n_units")
            var_shape = input_layer.tensor.get_shape().as_list()

            n_units = input_layer.n_units
            dtype = input_layer.dtype
        else:
            if n_units is not None and var_shape is not None:
                if var_shape[-1] != n_units:
                    raise ValueError("n_units {} does not match var_shape last dimension {}: ".format(n_units,
                                                                                                      var_shape[-1]))
            elif n_units is not None:
                var_shape = [1, n_units]

            if var_shape is None:
                raise ValueError("shape could not be determined: either supply an input layer or shape")
            else:
                n_units = var_shape[-1]

        if var_shape[0] is None or isinstance(var_shape[0], tf.Tensor):
            var_shape[0] = 1

        self.var_shape = var_shape

        if n_units is None:
            raise ValueError("invalid variable layer parameters: either supply input layer or a valid shape")

        # if input_layer is not None:
        #    shape = [input_layer.n_units, input_layer.n_units]
        # else:
        #    shape = [None, var_shape[-1]]

        super().__init__(input_layer, n_units, dtype=dtype, name=name)

        self.trainable = trainable
        self.resource = resource
        self.init = init if init is not None else zeros_init(dtype=self.dtype)
        self.tensor = self._build_graph()

    def _build_graph(self):
        var_reuse = self.share_vars_with is not None
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        input_layer = self.input_layers[-1] if len(self.input_layers) != 0 else None

        var_shape = self.var_shape

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):

            # ResourceVariable doesn't have a set_shape
            if input_layer is not None:
                init_shape = [1] + var_shape[1:] if var_shape[0] is None else var_shape
                validate_shape = False
            else:
                init_shape = var_shape
                validate_shape = True

            if self.share_vars_with is None:
                self.variable = tf.get_variable(self.name + "_var",
                                                shape=init_shape,
                                                dtype=self.dtype,
                                                trainable=self.trainable,
                                                validate_shape=validate_shape,
                                                initializer=self.init,
                                                use_resource=self.resource)

                self.counter = tf.get_variable(name="counter",
                                               shape=[],
                                               initializer=zeros_init(dtype=tf.int32),
                                               trainable=False,
                                               use_resource=True)
            else:
                self.variable = self.share_vars_with.variable
                self.counter = self.share_vars_with.counter

            if not self.resource and var_shape[0] is None:
                self.variable.set_shape(tf.TensorShape(self.var_shape))

            if input_layer is not None:

                def update_fn():

                    inc_counter = tf.assign_add(self.counter, 1)
                    with tf.control_dependencies([inc_counter]):
                        update_var = tf.assign(self.variable, input_layer.tensor, validate_shape=False)
                    return update_var

                # assign from input mode
                if self.init_from_input:
                    update = tf.cond(tf.math.less(self.counter, 1),
                                     update_fn,
                                     lambda: self.variable)

                    # control flow erases the shape of our tensor
                    # update = tf.reshape(update, [-1, self.n_units])
                    update.set_shape([None, self.n_units])
                else:
                    update = update_fn()
                tensor = update
            else:
                tensor = self.variable

            self._add_variable(self.variable)

            return tensor

    def reset(self):
        """ reset

        resets the variable using its initializer

        Returns:
            an op that can be run to reinitialize the variable
        """
        return tf.group(self.variable.initializer, self.counter.initializer)

    def reuse_with(self, input_layer=None, init_from_input=None, name=None):
        input_layer = self.input_layers[0] if input_layer is None else input_layer
        name = self.name if name is None else name
        init_from_input = self.init_from_input if init_from_input is None else init_from_input

        return VariableLayer(input_layer=input_layer,
                             var_shape=self.var_shape,
                             init_from_input=init_from_input,
                             trainable=self.trainable,
                             resource=self.resource,
                             dtype=self.dtype,
                             init=self.init,
                             share_vars_with=self,
                             name=name
                             )


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
        inputs = as_list(inputs)

        graph, endpoints = _get_subgraph(inputs, output)
        self.graph = graph
        self.end_points = endpoints
        self.output = output

        super().__init__(input_layers=inputs,
                         n_units=output.n_units,
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
        if len(layers) < 2:
            raise ValueError("Compose requires at least two layers")

        self.layers = layers

        in_layer, *others = layers
        super().__init__(input_layers=self.layers[0].input_layers,
                         n_units=others[-1].n_units,
                         dtype=self.layers[-1].dtype,
                         name=name)

        for curr_layer in others:
            if in_layer not in curr_layer.input_layers:
                raise ValueError("\n Invalid Compose: \n {} --x {} \n [not connected]".format(in_layer, curr_layer))
            in_layer = curr_layer

        # forward any feedable layer from the first layer
        if hasattr(self.layers[0], "placeholder"):
            self.placeholder = self.layers[0].placeholder

        # add both layer variables to the current layer container
        for var in itertools.chain.from_iterable([layer.variables for layer in self.layers]):
            self._add_variable(var)

        self.tensor = self.layers[-1].tensor

    def reuse_with(self, *input_layers, name=None):
        if name is None:
            name = self.name

        layer1, *layers = self.layers

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

        return Compose(layer1, *new_layers, name=name)

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

    def __init__(self, n_units, n_active=None, shape=None, batch_size=None, value=None, dtype=tf.float32,
                 name="input"):
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

        super().__init__(None, n_units, dtype, name)

        if shape is None:
            shape = [None, n_units]
        else:
            if shape[-1] != n_units:
                raise ValueError("Shape mismatch: shape[-1] ({}) != n_units ({})".format(shape[-1], n_units))

        self.value = value

        with layer_scope(self):
            # if n_active is not None convert to tf.SparseTensor
            if n_active is None:
                self.placeholder = tf.placeholder(dtype=self.dtype, shape=shape, name=self.name)
                self.tensor = self.placeholder
            else:  # sparse
                self.n_active = n_active
                self.placeholder = tf.placeholder(dtype=self.dtype, shape=[batch_size, n_active], name=self.name)

                self.tensor = transform.sparse_one_hot(self.placeholder, num_cols=self.n_units, dtype=self.dtype)

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

    def reuse_with(self, *layers, name=None):
        raise AttributeError("Cannot call reuse_with on Input Layer: Input has no input layers")

    def eval(self, feed_dict=None, session=None):
        if feed_dict is not None:
            if self.placeholder not in feed_dict:
                raise ValueError("feed dict does not contain the placeholder in this Input Layer")
        elif self.value is not None:
            feed_dict = {self.placeholder: self.value}
        return self.tensor.eval(feed_dict, session)


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

    def __init__(self, n_units, shape=None, dtype=tf.float32, value=None, name="sparse_input"):
        super().__init__(None, n_units, dtype, name)
        self.value = value

        if shape is None:
            shape = [None, n_units]
        else:
            if shape[-1] != n_units:
                raise ValueError("Shape mismatch: shape[-1] ({}) != n_units ({})".format(shape[-1], n_units))

        with layer_scope(self):
            self.placeholder = tf.sparse_placeholder(dtype, shape=shape, name=name)

            self.tensor = self.placeholder

    def reuse_with(self, *layers, name=None):
        raise AttributeError("Cannot call reuse_with on SparseInput Layer: SparseInput has no input layers")

    def eval(self, feed_dict=None, session=None):
        if feed_dict is not None:
            if self.placeholder not in feed_dict:
                raise ValueError("feed dict does not contain the placeholder in this Input Layer")
        elif self.value is not None:
            feed_dict = {self.placeholder: self.value}
        return self.tensor.eval(feed_dict, session)


class TensorLayer(Layer):
    """ Tensor Input Layer

    Attributes:
        tensor: the tensor to be wrapped by this layer
        var_list: if vars are involved in the output tensor, they can be specified here
        and will be listed in variables
        n_units: number of units for this layer,
        batch_size: Optional batch size for this layer

    Creates a layer from a given tensor that one can then integrate with other layers
    """

    def __init__(self, tensor, n_units=None, var_list=None, dtype=None, name="tensor_layer"):
        try:
            tensor = tx_utils.to_tensor_cast(tensor, dtype)
        except ValueError:
            raise ValueError("Could not convert tensor param with value {} to Tensor or tf.SparseTensor".format(tensor))

        dtype = tensor.dtype

        output_shape = tensor.get_shape().as_list()
        self.output_shape = output_shape
        shape = [output_shape[-1]]

        if n_units is None:
            n_units = output_shape[-1]

        super().__init__(input_layers=None, n_units=n_units, dtype=dtype, name=name)

        if var_list is not None:
            for var in var_list:
                self._add_variable(var)

        self.tensor = tensor

    def reuse_with(self, *layers, name=None):
        raise AttributeError("Cannot call reuse_with on TensorLayer Layer: TensorLayer has no input layers")


class LambdaLayer(Layer):
    """ Custom Fn Layer

    Attributes:
        tensor: the tensor to be wrapped by this layer
        var_list: if vars are involved in the output tensor, they can be specified here
        and will be listed in variables
        n_units: number of units for this layer,
        batch_size: Optional batch size for this layer
        layer_fn: if False applies the function to the tensors otherwise applies to the layer

    Creates a layer from a given tensor that one can then integrate with other layers
    """

    def __init__(self, *layers, apply_fn, n_units=None, var_list=None, dtype=None,
                 name="fn_layer",
                 layer_fn=False):
        self.apply_fn = apply_fn
        self.var_list = var_list
        self.layer_fn = layer_fn

        with layer_scope(self, name=name):
            fn_inputs = [layer.tensor if not layer_fn else layer for layer in layers]
            tensor = apply_fn(*fn_inputs)
            if dtype is not None and tensor.dtype != dtype and not isinstance(tensor, tf.Operation):
                tensor = tf.cast(tensor, dtype)
            dtype = tensor.dtype if not isinstance(tensor, tf.Operation) else None

        output_shape = tensor.get_shape().as_list()
        self.output_shape = output_shape

        if n_units is None:
            n_units = output_shape[-1] if len(output_shape) > 0 else 1

        if var_list is not None:
            for var in var_list:
                self._add_variable(var)

        super().__init__(input_layers=layers,
                         n_units=n_units,
                         dtype=dtype,
                         name=name)

        output_shape = [s if s is not None else -1 for s in output_shape]
        self.tensor = tf.reshape(tensor, output_shape)

    def reuse_with(self, *layers, name=None):
        return LambdaLayer(*layers,
                           apply_fn=self.apply_fn,
                           n_units=self.n_units,
                           var_list=self.var_list,
                           dtype=self.dtype,
                           name=self.name,
                           layer_fn=self.layer_fn)


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
                 input_layer,
                 n_units,
                 weight_init=glorot_uniform(),
                 shared_weights=None,
                 shared_bias=None,
                 transpose_weights=False,
                 sparse_weights=False,
                 add_bias=True,
                 bias_init=zeros_init(),
                 dtype=tf.float32,
                 name="linear", share_vars_with=None):

        self.shared_weights = shared_weights
        self.shared_bias = shared_bias
        self.weight_init = weight_init
        self.add_bias = add_bias
        self.share_vars_with = share_vars_with
        self.transpose_weights = transpose_weights
        self.sparse_weights = sparse_weights
        self.bias_init = bias_init

        if not isinstance(input_layer, Layer):
            input_layer = TensorLayer(input_layer, dtype=dtype)

        if input_layer.n_units is None or isinstance(input_layer.n_units, tf.Tensor):
            raise ValueError("Cannot create Linear layer from unknown previous n_units")

        super().__init__(input_layers=input_layer, n_units=n_units, dtype=dtype, name=name)

        if self.share_vars_with is not None:
            if not isinstance(self.share_vars_with, Linear):
                raise TypeError("Layer can only share variables with other layer of the same type")

            shape = [input_layer.n_units, self.n_units]
            shared_shape = self.share_vars_with.weights.get_shape().as_list()
            if self.transpose_weights:
                shared_shape = shared_shape[::-1]

            if shape != shared_shape:
                raise ValueError("Can only share variables with layers with the same dimensions: "
                                 "share_vars_with is provided but \n"
                                 "self shape: {s0} different from "
                                 "other shape: {s1}".format(s0=shape, s1=shared_shape))

        # if weights are passed, check that their shape matches the layer shape
        if self.shared_weights is not None:
            weight_shape = self.shared_weights.get_shape()

            if self.transpose_weights:
                if not tf.TensorShape([input_layer.n_units]).is_compatible_with(tf.TensorShape([weight_shape[-1]])):
                    raise ValueError(
                        "weight shape mismatch: input_layer shape {} :: weights shape {} with transpose_weights=True".format(
                            input_layer.shape,
                            weight_shape))
            else:
                if not tf.TensorShape([input_layer.n_units]).is_compatible_with(tf.TensorShape([weight_shape[0]])):
                    raise ValueError(
                        "weight shape mismatch: input_layer shape {} :: weights shape {} with transpose_weights=False".format(
                            input_layer.shape,
                            weight_shape))
        if self.shared_bias is not None:
            bias_shape = self.shared_bias.get_shape().as_list()
            if bias_shape[0] != self.n_units:
                raise ValueError(
                    "invalid shared bias: number of bias {} does not match number of units {}".format(bias_shape[0],
                                                                                                      self.n_units))

        self.tensor = self._build_graph()

    def _build_graph(self):
        input_layer = self.input_layers[0]
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with tf.name_scope(name) as scope, variable_scope.variable_scope(scope[:-1]):
            # init weights

            if self.shared_weights is None:
                if self.share_vars_with is None:
                    shape = [input_layer.n_units, self.n_units]
                    self.weights = tf.get_variable("weights",
                                                   shape=shape,
                                                   dtype=self.dtype,
                                                   use_resource=True,
                                                   initializer=self.weight_init)
                else:
                    self.weights = self.share_vars_with.weights
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
                    dense_sp = tf.sparse.to_dense(sp_values)
                    lookup_sum = tf.math.sparse_matmul(a=dense_sp,
                                                       b=self.weights,
                                                       a_is_sparse=True,
                                                       b_is_sparse=self.sparse_weights,
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
                rank = len(input_layer.tensor.get_shape().as_list())
                if rank > 2:
                    if self.transpose_weights:
                        axes = [[rank - 1], [1]]
                    else:
                        axes = [[rank - 1], [0]]
                    # Broadcasting is required for the inputs.
                    tensor = tf.tensordot(a=input_layer.tensor,
                                          b=self.weights,
                                          axes=axes)
                    # Reshape the output back to the original ndim of the input.
                    if not tf.executing_eagerly():
                        shape = input_layer.tensor.get_shape().as_list()
                        output_shape = shape[:-1] + [self.n_units]
                        tensor.set_shape(output_shape)
                else:
                    tensor = tf.matmul(a=input_layer.tensor,
                                       b=self.weights,
                                       name="mat_mul",
                                       transpose_b=self.transpose_weights,
                                       b_is_sparse=self.sparse_weights)

            # y = xW + [b]
            self.bias = None
            if self.shared_bias is None:
                if self.add_bias and self.n_units is not None:
                    if self.share_vars_with is None:
                        self.bias = tf.get_variable("bias",
                                                    shape=[self.n_units],
                                                    dtype=self.dtype,
                                                    initializer=self.bias_init,
                                                    use_resource=True)
                    else:
                        self.bias = self.share_vars_with.bias
            else:
                self.bias = self.shared_bias

            if self.bias is not None:
                self._add_variable(self.bias)
                tensor = tf.nn.bias_add(tensor, self.bias, name="add_b")

        return tensor

    def reuse_with(self, input_layer, name=None, transpose_weights=None, sparse_weights=None):
        """ Reuses the current layer on a different input.

        Uses the variables in this layer to create a new Layer instance with a different input_layer

        Args:
            input_layer: a ``Linear` layer
            name: name for the new ``Layer``

        Return:
            ``Layer``: a new layer with shared variables with the current layer.

        """
        # if current layer is sharing variables, forward the sharing
        share_vars_with = self if self.share_vars_with is None else self.share_vars_with

        if name is None:
            name = self.name

        if transpose_weights is None:
            transpose_weights = self.transpose_weights
        if sparse_weights is None:
            sparse_weights = self.sparse_weights

        return Linear(input_layer=input_layer,
                      n_units=self.n_units,
                      weight_init=self.weight_init,
                      shared_weights=self.shared_weights,
                      transpose_weights=transpose_weights,
                      sparse_weights=sparse_weights,
                      add_bias=self.add_bias,
                      name=name,
                      share_vars_with=share_vars_with)


class DropLookup(Layer):
    """ Applies Dropout with a given probability to a Lookup layer by unique id

    The lookup is expected to represent a sequence lookup in batch-major form
    For each lookup sample, the dropout is applied per unique id, meaning that if
    the lookup ids in a sample are [1,2,1] and id 1 is selected for dropout,
    the first and last vectors will be set to 0 for this sample.

    Args:
        layer: a Lookup input layer
        scale: if scale is True, scales the non-dropped lookups by x / (1 - probability)


    """

    def __init__(self, layer, probability=0.5, scale=True, name="drop_lookup"):
        if not isinstance(layer, Lookup):
            raise TypeError("input layer should be a {} layer {} found instead".format(str(Lookup), type(layer)))

        self.scale = scale
        self.probability = probability
        super().__init__(input_layers=layer, n_units=layer.n_units, name=name)

        self.tensor = self._build_graph()

    def _build_graph(self):
        if self.probability == 1:
            return tf.zeros_like(self.input_layers[0].tensor, dtype=tf.float32)
        elif self.probability > 0:

            input_indices = self.input_layers[0].input_layers[0]
            inputs = input_indices.tensor
            lookup_shape = tf.shape(self.input_layers[0].tensor)
            batch_size, seq_size = lookup_shape[0], lookup_shape[1]

            if input_indices.is_sparse():
                _, ids = tf.split(inputs.indices, 2, axis=-1)
            else:
                ids = inputs

            unique_ids, indices = tf.unique(tf.reshape(ids, [-1]))
            mask_shape = tf.stack([batch_size, tf.size(unique_ids)])
            unique_mask = tf.random_uniform(mask_shape, dtype=tf.float32)

            batch_wise = tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=-1),
                                         tf.stack([batch_size, seq_size]))
            unique_batch_wise = tf.reshape(indices, [batch_size, seq_size])

            # gather mask and convert it to binary mask
            mask_indices = tf.stack([batch_wise, unique_batch_wise], axis=-1)
            binary_mask = tf.floor(tf.gather_nd(unique_mask, mask_indices) + (1 - self.probability))
            if self.scale:
                binary_mask /= (1 - self.probability)

            dropped_lookup = self.input_layers[0].tensor * tf.expand_dims(binary_mask, axis=-1)
        else:
            dropped_lookup = self.input_layers[0].tensor

        return dropped_lookup


class ViewLayer(Layer):
    """ ViewLayer

    Has same shape and inputs as input layer and stores this layer for future reference.
    This means ViewLayer can substitute Layer where layer would be used

    Properties:
        inner_layer (Layer) wrapped by a view
    """

    def __init__(self, layer, dtype=None, attr_fwd=None, name=None):
        name = "view_{}".format(layer.name) if name is None else name
        dtype = layer.dtype if dtype is None else dtype
        self.inner_layer = layer
        super().__init__(input_layers=layer.input_layers,
                         n_units=layer.n_units,
                         dtype=dtype,
                         name=name)

        self.attr_fwd = as_list(attr_fwd)
        for attr in self.attr_fwd:
            if hasattr(self.inner_layer, attr):
                setattr(self, attr, getattr(self.inner_layer, attr))

        for var in self.inner_layer.variables:
            self._add_variable(var)


class DropConnect(ViewLayer):
    """ DropConnect

    Args:
            layer (Layer):
            keep_prob (float):
            weight_mask (Tensor):
            bias_mask (Tensor):
            locked (bool):
            name (str):

    """

    def __init__(self, layer, probability=0.5, weight_mask=None, bias_mask=None, locked=True, name=None):
        if not isinstance(layer, Linear):
            raise TypeError("DropConnect can only wrap Linear layers: {} found instead".format(layer))

        self.probability = probability
        self.weight_mask = weight_mask
        self.bias_mask = bias_mask
        self.bias = None
        self.weights = None
        self.locked = locked

        name = name if name is not None else "drop_{}".format(layer.name)

        super().__init__(layer, name=name)
        self.tensor = self._build_graph(layer.input_layers[0])

    def _build_graph(self, input_layer):
        with layer_scope(self):
            w = self.inner_layer.weights
            b = self.inner_layer.bias

            drop_w, w_mask = transform.dropout(w, probability=self.probability, random_mask=self.weight_mask,
                                               scale=False,
                                               return_mask=True)
            self.weight_mask = w_mask
            drop_b = None
            if b is not None:
                drop_b, b_mask = transform.dropout(b, probability=self.probability, random_mask=self.bias_mask,
                                                   scale=False,
                                                   return_mask=True)
                self.bias_mask = b_mask

            new_linear = Linear(input_layer, n_units=self.n_units, shared_weights=drop_w, shared_bias=drop_b)
            # forward weights and bias
            self.weights = new_linear.weights
            self.bias = new_linear.bias

            return new_linear.tensor

    def reuse_with(self, layer, name=None, locked=None):
        new_layer = self.inner_layer.reuse_with(layer)

        locked = self.locked if locked is None else locked
        name = self.name if name is None else name
        weight_mask = self.weight_mask if locked else None
        bias_mask = self.bias_mask if locked else None

        return DropConnect(layer=new_layer,
                           probability=self.probability,
                           weight_mask=weight_mask,
                           bias_mask=bias_mask,
                           locked=locked,
                           name=name)


class Dropout(Layer):
    """ Dropout

    Sets output units of the input layer to zero with a given probability and re-scales the remaining units to maintain
    the expected activation value.

    With probability ``keep_prob``, outputs the input elements scaled up by ``1 / keep_prob``, otherwise
    outputs ``0``. The scaling is to that the expected sum of the input elements is unchanged.

    Dropout can be viewed a stochastic version of model averaging and prevents the nodes from co-adapting too much. This
    reduces generalisation error during training.

    Args:
        layer: an input layer :class:`Layer` to which dropout will be applied
        keep_prob: a scalar float with the probability that each element is kept.
        seed: A Python integer. Used to create a random seed for the dropout op.

    Warning:
        if input is sparse the noise shape is not used

    References:
        [1] "Dropout:  A Simple Way to Prevent Neural Networks from Overfitting"
        http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, input_layer,
                 probability=0.1,
                 scale=True,
                 noise_shape=None,
                 mask=None,
                 locked=False,
                 seed=None,
                 name="dropout"):
        input_layer = convert_to_layer(input_layer)
        self.seed = seed
        self.probability = probability
        self.scale = scale
        self.noise_shape = noise_shape
        self.locked = locked
        self.mask = mask

        if self.mask is not None:
            if not isinstance(self.mask, Layer):
                self.mask = tx_utils.to_tensor_cast(self.mask)

        if self.noise_shape is not None:
            input_shape = tf.shape(input_layer.tensor)
            self.noise_shape = [input_shape[axis] if dim is None else dim for axis, dim in enumerate(self.noise_shape)]

        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=input_layer.dtype,
                         name=name)
        self.tensor = self._build_graph(input_layer)

    def _build_graph(self, layer):
        with layer_scope(self):
            if layer.is_sparse():
                # if input is sparse, noise_shape is not used
                tensor, mask = transform.sparse_dropout(sp_tensor=layer.tensor,
                                                        mask=self.mask,
                                                        probability=self.probability,
                                                        scale=self.scale,
                                                        return_mask=True,
                                                        seed=self.seed)

            else:
                tensor, mask = transform.dropout(tensor=layer.tensor,
                                                 noise_shape=self.noise_shape,
                                                 random_mask=self.mask,
                                                 probability=self.probability,
                                                 scale=self.scale,
                                                 return_mask=True,
                                                 seed=self.seed)

            if self.mask is None:
                self.mask = mask

            return tensor

    def reuse_with(self, layer, name=None, locked=None):
        locked = self.locked if locked is None else locked
        name = self.name if name is None else name
        mask = self.mask if locked else None

        return Dropout(layer,
                       probability=self.probability,
                       noise_shape=self.noise_shape,
                       mask=mask,
                       scale=self.scale,
                       locked=locked,
                       seed=self.seed,
                       name=name)


class FC(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 activation=identity,
                 weight_init=random_uniform(),
                 shared_weights=None,
                 transpose_weights=False,
                 add_bias=True,
                 bias_init=zeros_init(),
                 dtype=tf.float32,
                 name="fn",
                 share_vars_with=None):

        with layer_scope(self, name=name):
            self.linear = Linear(input_layer=layer,
                                 n_units=n_units,
                                 weight_init=weight_init,
                                 shared_weights=shared_weights,
                                 transpose_weights=transpose_weights,
                                 add_bias=add_bias,
                                 bias_init=bias_init,
                                 dtype=dtype,
                                 name="{}_linear".format(name),
                                 share_vars_with=share_vars_with)

            self.activation = Activation(self.linear, fn=activation, name="{}_activation".format(name))

        super().__init__(input_layers=layer,
                         n_units=self.activation.n_units,
                         dtype=self.activation.dtype,
                         name=name)

        for var in self.linear.variables:
            self._add_variable(var)
        self.tensor = self.activation.tensor

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name

        return FC(layer=layer,
                  n_units=self.n_units,
                  activation=self.activation.fn,
                  weight_init=self.linear.weight_init,
                  shared_weights=self.linear.shared_weights,
                  transpose_weights=self.linear.transpose_weights,
                  add_bias=self.linear.add_bias,
                  bias_init=self.linear.bias_init,
                  dtype=self.activation.dtype,
                  name=name,
                  share_vars_with=self.linear)


def _conv_output_length(input_length, kernel_size, padding, stride, dilation=1):
    if input_length is None:
        return None
    assert padding in {'SAME', 'VALID', 'CAUSAL'}
    dilated_filter_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    if padding == 'SAME':
        output_length = input_length
    elif padding == 'VALID':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'CAUSAL':
        output_length = input_length
    return (output_length + stride - 1) // stride


def _conv_out_shape(input_shape, filter_shape, padding, stride, dilation_rate):
    stride = as_list(stride)
    dilation_rate = as_list(dilation_rate)

    space = input_shape[1:-1]
    new_space = []
    for i in range(len(space)):
        new_dim = _conv_output_length(
            space[i],
            filter_shape[i],
            padding=padding,
            stride=stride[i],
            dilation=dilation_rate[i])
        new_space.append(new_dim)
    return (input_shape[0],) + tuple(new_space) + (filter_shape[-1],)


class Conv1D(Layer):
    """1D Convolution

    Assumes the input to have a shape [b,s,n]
    produces an output of shape [b,s,m] where m is the number of filters

    Args:
        layer: the input layer to the Convolution Layer
        n_units: number of output units for this layer (number of filters)
        filter_size: convolution filter size

    """

    def __init__(self, layer,
                 n_units,
                 filter_size,
                 stride=1,
                 dilation_rate=1,
                 same_padding=True,
                 init=random_uniform(),
                 use_bias=True,
                 name="conv1D",
                 share_state_with=None,
                 shared_filters=None):

        self.same_padding = same_padding
        self.dilation_rate = dilation_rate
        self.stride = stride
        self.filter_size = filter_size
        self.init = init
        self.use_bias = use_bias
        self.filter_shape = [self.filter_size, layer.n_units, n_units]
        self.share_state_with = share_state_with
        self.shared_filters = shared_filters
        self.padding = "SAME" if same_padding else "VALID"

        input_tensor_shape = layer.tensor.get_shape()
        output_shape = _conv_out_shape(input_tensor_shape, self.filter_shape, self.padding, stride, dilation_rate)
        self.output_shape = tf.TensorShape(output_shape).as_list()
        shape = [layer.n_units, n_units]

        if self.shared_filters is not None:
            if not self.shared_filters.get_shape().is_compatible_with(tf.TensorShape(self.filter_shape)):
                raise ValueError(
                    "invalid shared kernel weight shape: {} != expected :{}".format(
                        self.shared_filters.get_shape().as_list(),
                        self.filter_shape))

        if self.share_state_with is not None:
            if not isinstance(self.share_state_with, Conv1D):
                raise TypeError("Layer can only share variables with other layer of the same type")

            if self.filter_shape != self.share_state_with.filter_shape:
                raise ValueError("Can only share variables between layers with the same kernel shape: \n"
                                 "Current layer: {}\n"
                                 "Shared Weights from: {}".format(self.filter_shape,
                                                                  self.share_state_with.filter_shape)
                                 )

        super().__init__(input_layers=layer, n_units=n_units, dtype=tf.float32, name=name)

        self.tensor = self._build_graph(layer)

    def _build_graph(self, layer):
        var_scope_name = self.share_state_with.scoped_name if self.share_state_with is not None else None
        var_reuse = self.share_state_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with tf.name_scope(name) as scope, variable_scope.variable_scope(scope[:-1]):
            # init weights

            if self.shared_filters is None:
                self.filters = tf.get_variable("filters",
                                               shape=self.filter_shape,
                                               dtype=self.dtype,
                                               initializer=self.init,
                                               use_resource=True)
            else:
                self.filters = self.shared_filters

            # store variables for easy access
            self._add_variable(self.filters)
            # y = conv1D(x,w)
            if layer.is_sparse():
                input_tensor = tf.sparse.to_dense(layer.tensor)
            else:
                input_tensor = layer.tensor

            if input_tensor.dtype == tf.float64:
                input_tensor = tf.cast(input_tensor, tf.float32)

            tensor = tf.nn.convolution(input=input_tensor,
                                       filter=self.filters,
                                       padding=self.padding,
                                       strides=(self.stride,),
                                       dilation_rate=(self.dilation_rate,),
                                       data_format="NWC")

            # y = xW + [b]
            if self.use_bias:
                self.bias = tf.get_variable("bias",
                                            shape=[self.n_units],
                                            dtype=self.dtype,
                                            initializer=zeros_init(),
                                            use_resource=True)
                self._add_variable(self.bias)
                tensor = tf.nn.bias_add(tensor, self.bias, name="add_b")
        return tensor

    def reuse_with(self, layer, name=None):
        share_state_with = self if self.share_state_with is None else self.share_state_with
        if name is None:
            name = self.name

        return Conv1D(layer,
                      n_units=self.n_units,
                      filter_size=self.filter_size,
                      stride=self.stride,
                      dilation_rate=self.dilation_rate,
                      same_padding=self.same_padding,
                      init=self.init,
                      use_bias=self.use_bias,
                      name=name,
                      share_state_with=share_state_with)


class CausalConv(Conv1D):
    def __init__(self, layer,
                 n_units,
                 filter_size,
                 stride=1,
                 dilation_rate=1,
                 init=random_uniform(),
                 use_bias=True,
                 name="CausalConv",
                 share_state_with=None,
                 shared_filters=None):
        def causal_padding(x):
            left_pad = dilation_rate * (filter_size - 1)
            padding = [[0, 0], [left_pad, 0], [0, 0]]
            return tf.pad(x, padding)

        padded_layer = WrapLayer(layer, wrap_fn=causal_padding, name="causal_padding")

        super().__init__(layer=padded_layer,
                         n_units=n_units,
                         filter_size=filter_size,
                         stride=stride,
                         dilation_rate=dilation_rate,
                         same_padding=False,
                         init=init,
                         use_bias=use_bias,
                         name=name,
                         share_state_with=share_state_with,
                         shared_filters=shared_filters)

    def reuse_with(self, layer, name=None):
        share_vars_with = self if self.share_state_with is None else self.share_state_with
        if name is None:
            name = self.name

        return CausalConv(layer,
                          n_units=self.n_units,
                          filter_size=self.filter_size,
                          stride=self.stride,
                          dilation_rate=self.dilation_rate,
                          init=self.init,
                          use_bias=self.use_bias,
                          name=name,
                          share_state_with=share_vars_with,
                          shared_filters=self.shared_filters)


class Conv2D(Layer):
    """2D Convolution

    Transforms a 3D tensor input into a 3D tensor output by convolving a parametric filter with the input

    The input should have the shape [batch, height, width, channels] (channels are always last)
    The output has the shape [batch, height, width, n_units] where n_units is the number of filters to be applied


    Notes on layers dimensions:
        If we have a stride of size 1 and we set the zero padding to::

            :math:`\text{Zero Padding}=\frac{K-1}{2}`

        where K is the filter size. Then the output and input will have the same spatial dimensions

        The output size of a filter size of K is::

            :math:`O = \frac{(W-K+2P)}{S}+1`

        where O is the output height/length, W is the input height/length, K is the filter size,
        P is the padding, and S is the stride.

    Notes on padding:

        VALID padding with filter of size 6:

        inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                      |________________|                dropped
                                     |_________________|


        SAME padding with filter size of 6:

                    pad|                                      |pad
        inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
               |________________|
                              |_________________|
                                             |________________|


    Args:
        layer: the input layer to the Convolution Layer
        n_units: number of output units for this layer (number of filters)
        filter_size: (h,w) int tuple/list or single value v that is expanded into a convolution filter of size (v,v)
        stride: (h,w) int tuple/list of single value v that is expanded into convolution stride (v,v)
        dilation_rate: (h,w) int tuple/list of single value v that is expanded into convolution stride (v,v)
        same_padding: if True uses the same padding, else uses valid where we use no padding and depending on the filter
        size, some elements can be dropped (see notes above)

    """

    def __init__(self, layer,
                 n_units,
                 filter_size,
                 stride=(1, 1),
                 dilation_rate=(1, 1),
                 same_padding=True,
                 init=random_uniform(),
                 bias=True,
                 name="conv2D",
                 share_vars_with=None,
                 shared_filters=None):

        self.same_padding = same_padding

        dilation_rate = as_list(dilation_rate)
        if len(dilation_rate) == 1:
            dilation_rate *= 2
        self.dilation_rate = dilation_rate

        stride = as_list(stride)
        if len(stride) == 1:
            stride *= 2
        self.stride = stride

        filter_size = as_list(filter_size)
        if len(filter_size) == 1:
            filter_size *= 2
        self.filter_size = filter_size

        self.init = init
        self.bias = bias

        self.filter_shape = self.filter_size + [layer.n_units, n_units]
        self.share_vars_with = share_vars_with
        self.shared_filters = shared_filters
        self.padding = "SAME" if same_padding else "VALID"

        input_tensor_shape = layer.tensor.get_shape()
        output_shape = _conv_out_shape(input_tensor_shape, self.filter_shape, self.padding, stride, dilation_rate)
        self.output_shape = tf.TensorShape(output_shape).as_list()
        shape = [layer.n_units, n_units]

        if self.shared_filters is not None:
            if not self.shared_filters.get_shape().is_compatible_with(tf.TensorShape(self.filter_shape)):
                raise ValueError(
                    "invalid shared kernel weight shape: {} != expected :{}".format(
                        self.shared_filters.get_shape().as_list(),
                        self.filter_shape))

        if self.share_vars_with is not None:
            if not isinstance(self.share_vars_with, Conv1D):
                raise TypeError("Layer can only share variables with other layer of the same type")

            if self.filter_shape != self.share_vars_with.filter_shape:
                raise ValueError("Can only share variables between layers with the same kernel shape: \n"
                                 "Current layer: {}\n"
                                 "Shared Weights from: {}".format(self.filter_shape,
                                                                  self.share_vars_with.filter_shape)
                                 )

        super().__init__(input_layers=layer, n_units=n_units, dtype=tf.float32, name=name)

        self.tensor = self._build_graph(layer)

    def _build_graph(self, layer):
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with tf.name_scope(name) as scope, variable_scope.variable_scope(scope[:-1]):
            # init weights

            if self.shared_filters is None:
                self.filters = tf.get_variable("filters",
                                               shape=self.filter_shape,
                                               dtype=self.dtype,
                                               initializer=self.init,
                                               use_resource=True)
            else:
                self.filters = self.shared_filters

            # store variables for easy access
            self._add_variable(self.filters)
            # y = conv1D(x,w)
            if layer.is_sparse():
                input_tensor = tf.sparse.to_dense(layer.tensor)
            else:
                input_tensor = layer.tensor

            if input_tensor.dtype == tf.float64:
                input_tensor = tf.cast(input_tensor, tf.float32)

            tensor = tf.nn.convolution(input=input_tensor,
                                       filter=self.filters,
                                       padding=self.padding,
                                       strides=self.stride,
                                       dilation_rate=self.dilation_rate,
                                       data_format="NHWC")

            # y = xW + [b]
            if self.bias:
                self.bias = tf.get_variable("bias",
                                            shape=[self.n_units],
                                            dtype=self.dtype,
                                            initializer=zeros_init(),
                                            use_resource=True)
                self._add_variable(self.bias)
                tensor = tf.nn.bias_add(tensor, self.bias, name="add_b")
        return tensor

    def reuse_with(self, layer, name=None):
        share_vars_with = self if self.share_vars_with is None else self.share_vars_with
        if name is None:
            name = self.name

        return Conv2D(layer,
                      self.n_units,
                      self.filter_size,
                      self.stride,
                      self.dilation,
                      self.same_padding,
                      self.bias,
                      name,
                      share_vars_with)


class QRNN(Layer):
    def __init__(self,
                 layer,
                 n_units,
                 activation=tanh,
                 filter_size=2,
                 stride=1,
                 dilation_rate=1,
                 init_candidate=random_uniform(),
                 init_forget_gate=random_uniform(),
                 output_gate=True,
                 init_output_gate=random_uniform(),
                 bias=True,
                 input_gate=False,
                 init_input_gate=random_uniform(),
                 share_vars_with=None,
                 zoneout=False,
                 name="qrnn"):
        # this is computed in the conv layers as well
        filter_shape = [filter_size, layer.n_units, n_units]
        input_tensor_shape = layer.tensor.get_shape()
        output_shape = _conv_out_shape(input_tensor_shape, filter_shape, "CAUSAL", stride, dilation_rate)
        self.output_shape = tf.TensorShape(output_shape).as_list()
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.filter_size = filter_size
        self.init_candidate = init_candidate
        self.init_forget_gate = init_forget_gate
        self.init_output_gate = init_output_gate
        self.bias = bias
        self.input_gate = input_gate
        self.output_gate = output_gate
        self.init_input_gate = init_input_gate
        self.zoneout = zoneout

        if share_vars_with is not None and not isinstance(share_vars_with, QRNN):
            raise TypeError("shared qrnn must be of type {} got {} instead".format(QRNN, type(share_vars_with)))
        self.share_vars_with = share_vars_with

        super().__init__(input_layers=layer, n_units=n_units, dtype=tf.float32, name=name)

        self.tensor = self._build_graph(layer)

    def _build_graph(self, layer):
        with layer_scope(self):
            if self.share_vars_with is None:
                # candidate convolution
                self.w_z = CausalConv(layer=layer,
                                      n_units=self.n_units,
                                      filter_size=self.filter_size,
                                      stride=self.stride,
                                      dilation_rate=self.dilation_rate,
                                      init=self.init_candidate, use_bias=True,
                                      name="candidate_conv")

                # forget gate weights
                self.w_f = CausalConv(layer=layer,
                                      n_units=self.n_units,
                                      filter_size=self.filter_size,
                                      stride=self.stride,
                                      dilation_rate=self.dilation_rate,
                                      init=self.init_forget_gate, use_bias=True,
                                      name="forget_conv")

                if self.input_gate:
                    self.w_i = CausalConv(layer=layer,
                                          n_units=self.n_units,
                                          filter_size=self.filter_size,
                                          stride=self.stride,
                                          dilation_rate=self.dilation_rate,
                                          init=self.init_input_gate, use_bias=True,
                                          name="input_conv")

                if self.output_gate:
                    self.w_o = CausalConv(layer=layer,
                                          n_units=self.n_units,
                                          filter_size=self.filter_size,
                                          stride=self.stride,
                                          dilation_rate=self.dilation_rate,
                                          init=self.init_output_gate, use_bias=True,
                                          name="output_conv")

            else:
                self.w_z = self.share_vars_with.w_z.reuse_with(layer)
                self.w_f = self.share_vars_with.w_f.reuse_with(layer)
                if self.output_gate:
                    self.w_o = self.share_vars_with.w_o.reuse_with(layer)

                if self.input_gate:
                    self.w_i = self.share_vars_with.w_i.reuse_with(layer)

            # add refs to vars that this layer uses
            layer_vars = []
            layer_vars.extend(self.w_z.variables)
            layer_vars.extend(self.w_f.variables)
            if self.input_gate:
                layer_vars.extend(self.w_i.variables)
            if self.output_gate:
                layer_vars.extend(self.w_o.variables)

            for var in layer_vars:
                self._add_variable(var)

            with tf.name_scope("pool"):
                input_batch = tf.shape(layer.tensor)[0]
                prev_candidate = tf.zeros([input_batch, self.n_units])
                prev_candidate = TensorLayer(prev_candidate, self.n_units)

                # as sequence views
                wz_seq = Transpose(self.w_z, [1, 0, 2])
                wf_seq = Transpose(self.w_f, [1, 0, 2])
                wo_seq = Transpose(self.w_o, [1, 0, 2])

                def forget_fn(x):
                    if self.zoneout:
                        return 1 - transform.dropout(1 - sigmoid(x), scale=False)
                    else:
                        return sigmoid(x)

                if self.input_gate:
                    wi_seq = Transpose(self.w_i, [1, 0, 2])

                states = []

                # TODO I'm not sure this is correct
                shape = [layer.n_units, self.n_units]
                for i in range(shape[1]):
                    wz_i = Activation(wz_seq[i], fn=self.activation, name="z_{}".format(i + 1))

                    if self.input_gate:
                        # independent input and forget gates
                        gated_prev = Gate(prev_candidate,
                                          gate_input=wf_seq[i],
                                          gate_fn=forget_fn,
                                          name="forget_gate_{}".format(i + 1))
                        gated_input = Gate(wz_i, gate_input=wi_seq[i],
                                           name="input_gate_{}".format(i + 1))

                        cur_candidate = Add(gated_prev, gated_input)
                    else:
                        # coupled input and forget gates
                        cur_candidate = CoupledGate(prev_candidate,
                                                    wz_i,
                                                    gate_input=wf_seq[i],
                                                    gate_fn=forget_fn,
                                                    name="forget_coupled_gate_{}".format(i + 1))
                    prev_candidate = cur_candidate

                    if self.output_gate:
                        states.insert(i, Gate(cur_candidate,
                                              gate_input=wo_seq[i],
                                              name="output_gate_{}".format(i + 1)))
                    else:
                        states.insert(i, cur_candidate)
                tensor = tf.stack([state.tensor for state in states])
                tensor = tf.transpose(tensor, [1, 0, 2])

        return tensor

    def reuse_with(self, layer, zoneout=False, name=None):
        share_vars_with = self if self.share_vars_with is None else self.share_vars_with
        if name is None:
            name = self.name

        return QRNN(layer=layer,
                    n_units=self.n_units,
                    activation=self.activation,
                    filter_size=self.filter_size,
                    stride=self.stride,
                    dilation_rate=self.dilation_rate,
                    init_candidate=self.init_candidate,
                    init_forget_gate=self.init_forget_gate,
                    init_output_gate=self.init_output_gate,
                    output_gate=self.output_gate,
                    input_gate=self.input_gate,
                    init_input_gate=self.init_input_gate,
                    bias=self.bias,
                    share_vars_with=share_vars_with,
                    zoneout=zoneout,
                    name=name)

    def as_seq(self):
        return WrapLayer(self, self.n_units, lambda x: tf.transpose(x, [1, 0, 2]),
                         attr_fwd=["w_z", "w_f", "w_o"])


class BaseRNNCell(Layer):
    """

    Args:
        input_layer the input Layer for this cell
        previous_state: the recurrent input Layer for the cell
        state_size: list of number of units for each element in the state, default is a single state with [n_units]
        n_units: number of activation units for the RNN cell
        dtype: Layer (output) dtype
    """

    @staticmethod
    def zero_state(state_shape, stateful=True, name="zero_state"):
        zero_state = TensorLayer(tf.zeros(state_shape), n_units=state_shape[-1], name=name)

        if stateful:
            # init only once from zero state
            zero_state = VariableLayer(input_layer=zero_state,
                                       n_units=state_shape[-1],
                                       init_from_input=True,
                                       name=name)

        return zero_state

    def __init__(self,
                 input_layer,
                 previous_state,
                 state_size,
                 n_units,
                 dtype=tf.float32,
                 w_init=glorot_uniform(),
                 u_init=glorot_uniform(),
                 activation=tanh,
                 w_dropconnect=None,
                 u_dropconnect=None,
                 x_dropout=None,
                 r_dropout=None,
                 y_dropout=None,
                 dropout_locked=True,
                 regularized=False,
                 share_state_with=None,
                 name="recurrent_cell"):
        if share_state_with is not None and not isinstance(share_state_with, type(self)):
            raise TypeError(
                "share_state_with must be of type {} got {} instead".format(type(self), type(share_state_with)))
        self.share_state_with = share_state_with
        self.dropout_locked = dropout_locked

        if state_size is None:
            state_size = [n_units]

        batch_size = tf.shape(input_layer.tensor)[0]

        def init_states(enum_state):
            i, state = enum_state
            if state is None:
                state_shape = [batch_size, state_size[i]]
                return BaseRNNCell.zero_state(state_shape)
            else:
                if isinstance(state, Layer):
                    if state.n_units != state_size[i]:
                        raise ValueError(
                            "previous state {i} n_units {n_units} not compatible with state shape {state_shape}".format(
                                i=i,
                                n_units=state.n_units,
                                state_shape=state_size[i]))
                else:
                    state = TensorLayer(state, n_units=state_size[i])
                return state

        if previous_state is not None:
            previous_state = as_list(previous_state)
            if len(previous_state) != len(state_size):
                raise ValueError(
                    "previous state should have {} states: {} passed instead".format(len(state_size),
                                                                                     len(previous_state)))
        else:
            previous_state = tuple([None] * len(state_size))
        # fills in all previous states
        previous_state = list(map(init_states, enumerate(previous_state)))

        self.previous_state = previous_state
        self.regularized = regularized
        self.w_dropconnect = w_dropconnect
        self.u_dropconnect = u_dropconnect
        self.x_dropout = x_dropout
        self.r_dropout = r_dropout
        self.y_dropout = y_dropout
        self.w_init = w_init
        self.u_init = u_init
        self.activation = activation

        # util class to apply regularizers or forward views
        class Regularizer:

            def __init__(self, func: LayerProto):
                self.func = func
                self.reg = None

            def reset(self):
                self.reg = None

            def __call__(self, *layers):

                reg_layers = []
                if self.reg is None:
                    self.reg = []
                    for layer in layers:
                        reg_layer = self.func(layer)
                        reg_layers.append(reg_layer)

                        if issubclass(self.func.layer_cls, ViewLayer):
                            self.reg.append(lambda x: x)
                        else:
                            # dropouts we can re-use
                            self.reg.append(reg_layer.reuse_with)
                else:
                    for layer, reg in zip(layers, self.reg):
                        reg_layers.append(reg(layer))

                if len(reg_layers) == 1:
                    return reg_layers[0]
                else:
                    return reg_layers

        # stores regularizers
        if self.share_state_with is None:
            self.w_reg = Regularizer(
                DropConnect.proto(probability=self.w_dropconnect, locked=True, name="w_dropconnect"))
            self.u_reg = Regularizer(
                DropConnect.proto(probability=self.u_dropconnect, locked=True, name="u_dropconnect"))
            self.x_reg = Regularizer(
                Dropout.proto(probability=self.x_dropout, locked=self.dropout_locked, name="x_dropout"))
            self.r_reg = Regularizer(
                Dropout.proto(probability=self.r_dropout, locked=self.dropout_locked, name="r_dropout"))
            self.y_reg = Regularizer(
                Dropout.proto(probability=self.y_dropout, locked=self.dropout_locked, name="y_dropout"))
        else:
            self.w_reg = self.share_state_with.w_reg
            self.u_reg = self.share_state_with.u_reg
            self.x_reg = self.share_state_with.x_reg
            self.r_reg = self.share_state_with.r_reg
            self.y_reg = self.share_state_with.y_reg

            if not self.regularized:
                self.w_reg.reset()
                self.u_reg.reset()
                self.x_reg.reset()
                self.r_reg.reset()
                self.y_reg.reset()

        # needs to be defined on each recurrent cell just as we define self.tensor
        # the default state is the current cell which gives access to its  output tensor
        self.state = self

        super().__init__(input_layers=[input_layer] + self.previous_state,
                         n_units=n_units,
                         dtype=dtype,
                         name=name)

    def reuse_with(self, input_layer, previous_state=None, regularized=None, name=None, **kwargs):
        # because we use objects and not scopes we can use self always on share state with
        share_state_with = self  # self if self.share_state_with is None else self.share_state_with
        previous_state = self.previous_state if previous_state is None else previous_state
        name = self.name if name is None else name
        regularized = self.regularized if regularized is None else regularized

        return type(self)(
            input_layer=input_layer,
            n_units=self.n_units,
            previous_state=previous_state,
            activation=self.activation,
            share_state_with=share_state_with,
            w_dropconnect=self.w_dropconnect,
            u_dropconnect=self.u_dropconnect,
            x_dropout=self.x_dropout,
            r_dropout=self.r_dropout,
            y_dropout=self.y_dropout,
            dropout_locked=self.dropout_locked,
            regularized=regularized,
            name=name,
            **kwargs
        )


class RNNCell(BaseRNNCell):
    """ Recurrent Cell
        Corresponds to a single step on an unrolled RNN network

        Args:
                input_layer: the input layer to the RNN Cell
                n_units: number of output units for this RNN Cell
                previous_state: a RNNCell from which we can extract output
                activation: activation function to be used in the cell
                share_state_with: a ``Layer`` with the same number of units than this Cell
                name: name for the RNN cell
        """

    def __init__(self,
                 input_layer,
                 n_units,
                 previous_state=None,
                 activation=tanh,
                 w_init=glorot_uniform(),
                 u_init=glorot_uniform(),
                 share_state_with=None,
                 w_dropconnect=None,
                 u_dropconnect=None,
                 r_dropout=None,
                 x_dropout=None,
                 y_dropout=None,
                 dropout_locked=True,
                 regularized=False,
                 name="rnn_cell"):

        super().__init__(input_layer=input_layer,
                         previous_state=previous_state,
                         state_size=None,
                         n_units=n_units,
                         activation=activation,
                         dtype=tf.float32,
                         w_init=w_init,
                         u_init=u_init,
                         w_dropconnect=w_dropconnect,
                         u_dropconnect=u_dropconnect,
                         x_dropout=x_dropout,
                         r_dropout=r_dropout,
                         y_dropout=y_dropout,
                         dropout_locked=dropout_locked,
                         regularized=regularized,
                         share_state_with=share_state_with,
                         name=name)

        tensor, state = self._build_graph()
        self.tensor = tensor
        self.state = [state]

    def _build_graph(self):
        regularized = self.regularized
        input_layer = self.input_layers[0]
        previous_h = self.previous_state[0]

        with layer_scope(self):
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_h = self.r_reg(previous_h)

            if self.share_state_with is None:
                self.w = Linear(input_layer, self.n_units, add_bias=True, weight_init=self.w_init, name="w")
                self.u = Linear(previous_h, self.n_units, add_bias=False, weight_init=self.u_init, name="r_w")
            else:
                w = self.share_state_with.w
                u = self.share_state_with.u
                # this means the previous layer was regularized we want the inner layer
                # get inner state of dropconnect or other views
                if not self.regularized:
                    w = w.inner_layer if isinstance(w, ViewLayer) else w
                    u = u.inner_layer if isinstance(u, ViewLayer) else u

                self.w = w.reuse_with(input_layer)
                self.u = u.reuse_with(previous_h)

            if regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    self.w = self.w_reg(self.w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    self.u = self.u_reg(self.u)

            output = Add(self.w, self.u)
            output = Activation(output, self.activation)

            current_h = Module(inputs=[previous_h, input_layer],
                               output=output,
                               name=self.name + "_h")

            if regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            return output.tensor, current_h


class GRUCell(BaseRNNCell):
    """ Gated Recurrent Unit Cell.

        Performs a single step with a gated recurrent unit where. These units have two gates:
        The first defines how much do we use the values from the recurrent connection to predict the current state
        The second
    """

    def __init__(self, input_layer, n_units,
                 previous_state=None,
                 activation=tanh,
                 gate_activation=sigmoid,
                 w_init=glorot_uniform(),
                 u_init=glorot_uniform(),
                 u_dropconnect=None,
                 w_dropconnect=None,
                 x_dropout=None,
                 r_dropout=None,
                 y_dropout=None,
                 dropout_locked=True,
                 regularized=False,
                 share_state_with=None,
                 name="gru_cell"):

        super().__init__(input_layer=input_layer,
                         previous_state=previous_state,
                         state_size=None,
                         n_units=n_units,
                         activation=activation,
                         dtype=tf.float32,
                         w_init=w_init,
                         u_init=u_init,
                         w_dropconnect=w_dropconnect,
                         u_dropconnect=u_dropconnect,
                         x_dropout=x_dropout,
                         r_dropout=r_dropout,
                         y_dropout=y_dropout,
                         dropout_locked=dropout_locked,
                         regularized=regularized,
                         share_state_with=share_state_with,
                         name=name)

        self.gate_activation = gate_activation
        tensor, state = self._build_graph()
        self.tensor = tensor
        self.state = [state]

    def _build_graph(self):

        def get_inner(layers):
            inner = []
            for layer in layers:
                if isinstance(layer, ViewLayer):
                    inner.append(layer.inner_layer)
                else:
                    inner.append(layer)
            return inner

        input_layer = self.input_layers[0]
        # previous state is a single layer
        regularized = self.regularized
        previous_h = self.previous_state[0]

        with layer_scope(self):

            # its not obvious that we forward w_reg from share_state_with?
            # regularize inputs and recurrent inputs
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_h = self.r_reg(previous_h)

            if self.share_state_with is None:
                # reset gate
                # forget / reset bias init to one http://proceedings.mlr.press/v37/jozefowicz15.pdf
                self.w_r = Linear(input_layer, self.n_units, add_bias=True, bias_init=ones_init(), name="w_r")
                self.u_r = Linear(previous_h, self.n_units, add_bias=False, name="u_r")

                self.w_c = Linear(input_layer, self.n_units, add_bias=True, weight_init=self.w_init, name="w_c")
                self.u_c = Linear(previous_h, self.n_units, add_bias=False, weight_init=self.u_init, name="u_c")

                self.w_z = Linear(input_layer, self.n_units, add_bias=True, name="w_z")
                self.u_z = Linear(previous_h, self.n_units, add_bias=False, name="u_z")

                self.w = [self.w_r, self.w_c, self.w_z]
                self.u = [self.u_r, self.u_c, self.u_z]
            else:
                self.w = self.share_state_with.w
                self.u = self.share_state_with.u

                if not self.regularized:
                    self.w = get_inner(self.w)
                    self.u = get_inner(self.u)

                self.w = [wi.reuse_with(input_layer) for wi in self.w]
                self.u = [ui.reuse_with(previous_h) for ui in self.u]

                self.w_r, self.w_c, self.w_z = self.w
                self.u_r, self.u_c, self.u_z = self.u

            if regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    self.w = self.w_reg(*self.w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    self.u = self.u_reg(*self.u)

                self.w_r, self.w_c, self.w_z = self.w
                self.u_r, self.u_c, self.u_z = self.u

            r_u_c = Gate(self.u_c, Add(self.w_r, self.u_r), name="reset_gate")
            candidate = Activation(Add(self.w_c, r_u_c), fn=self.activation, name="candidate")
            output = CoupledGate(candidate, previous_h, Add(self.w_z, self.u_z), name="output")

            current_h = Module(inputs=[previous_h, input_layer],
                               output=output,
                               name=self.name + "_h")
            if regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            return output.tensor, current_h

    def reuse_with(self, input_layer, previous_state=None, regularized=None, name=None):
        return super().reuse_with(input_layer, previous_state, regularized, name, gate_activation=self.gate_activation)


class LSTMCell(BaseRNNCell):
    """ LSTM Cell

        Performs a single step with a gated recurrent unit where. These units have two gates:
        The first defines how much do we use the values from the recurrent connection to predict the current state
        The second

        Args:
            previous_state (tuple): (previous_h, previous_memory) where previous_h is the previous output for the cell from a
            a previous timestep or None if the current cell is the first step
            previous_memory is the memory state output for the previous cell or None if the current cell is the first step
    """

    def __init__(self, input_layer, n_units,
                 previous_state=None,
                 activation=tanh,
                 gate_activation=sigmoid,
                 forget_bias_init=ones_init(),
                 w_init=glorot_uniform(),
                 u_init=glorot_uniform(),
                 w_dropconnect=None,
                 u_dropconnect=None,
                 x_dropout=None,
                 r_dropout=None,
                 y_dropout=None,
                 dropout_locked=True,
                 regularized=False,
                 share_state_with=None,
                 name="lstm_cell"):

        super().__init__(input_layer=input_layer,
                         previous_state=previous_state,
                         state_size=[n_units] * 2,
                         n_units=n_units,
                         activation=activation,
                         dtype=tf.float32,
                         w_init=w_init,
                         u_init=u_init,
                         w_dropconnect=w_dropconnect,
                         u_dropconnect=u_dropconnect,
                         x_dropout=x_dropout,
                         r_dropout=r_dropout,
                         y_dropout=y_dropout,
                         dropout_locked=dropout_locked,
                         regularized=regularized,
                         share_state_with=share_state_with,
                         name=name)

        self.forget_bias_init = forget_bias_init
        self.gate_activation = gate_activation
        tensor, h, memory_state = self._build_graph()

        self.tensor = tensor
        self.state = (h, memory_state)

        for l in self.w:
            for v in l.variables:
                self._add_variable(v)
        for l in self.u:
            for v in l.variables:
                self._add_variable(v)

    def _build_graph(self):
        def get_inner(layers):
            inner = []
            for layer in layers:
                if isinstance(layer, ViewLayer):
                    inner.append(layer.inner_layer)
                else:
                    inner.append(layer)
            return inner

        # input layers = [input_layer, *state_layer]
        input_layer = self.input_layers[0]
        regularized = self.regularized
        previous_h, previous_memory = self.previous_state

        with layer_scope(self):
            # regularize recurrent
            # regularize inputs and recurrent inputs
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_h = self.r_reg(previous_h)

            # create new weights
            if self.share_state_with is None:
                # forget gate linear
                # http://proceedings.mlr.press/v37/jozefowicz15.pdf bias forget = 1
                self.w_f = Linear(input_layer, self.n_units, add_bias=True, bias_init=self.forget_bias_init, name="w_f")
                self.u_f = Linear(previous_h, self.n_units, add_bias=False, name="u_f")

                # input gate linear
                self.w_i = Linear(input_layer, self.n_units, add_bias=True, name="w_i")
                self.u_i = Linear(previous_h, self.n_units, add_bias=False, name="u_i")

                # candidate linear
                self.w_c = Linear(input_layer, self.n_units, add_bias=True, name="w_c")
                self.u_c = Linear(previous_h, self.n_units, add_bias=False, name="u_c")

                # output gate
                self.w_o = Linear(input_layer, self.n_units, add_bias=True, name="w_o")
                self.u_o = Linear(previous_h, self.n_units, add_bias=False, name="u_o")

                self.w = [self.w_f, self.w_i, self.w_c, self.w_o]
                self.u = [self.u_f, self.u_i, self.u_c, self.u_o]

            else:
                self.w = self.share_state_with.w
                self.u = self.share_state_with.u

                # get inner state of dropconnect or other views
                if not self.regularized:
                    self.w = get_inner(self.w)
                    self.u = get_inner(self.u)

                self.w = [wi.reuse_with(input_layer) for wi in self.w]
                self.u = [ui.reuse_with(previous_h) for ui in self.u]

                self.w_f, self.w_i, self.w_c, self.w_o = self.w
                self.u_f, self.u_i, self.u_c, self.u_o = self.u

            # apply regularizers to weights
            if regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    self.w = self.w_reg(*self.w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    self.u = self.u_reg(*self.u)

                self.w_f, self.w_i, self.w_c, self.w_o = self.w
                self.u_f, self.u_i, self.u_c, self.u_o = self.u

            with tf.name_scope("memory_forget"):
                gate_f = Add(self.w_f, self.u_f, name="add_f")
                memory_state = Gate(previous_memory, gate_f, gate_fn=self.gate_activation, name="gated_memory")

            with tf.name_scope("candidate_store"):
                gate_i = Add(self.w_i, self.u_i, name="candidate_gate")
                candidate = Activation(Add(self.w_c, self.u_c), fn=self.activation,
                                       name="candidate_activation")
                candidate = Gate(candidate, gate_i, gate_fn=self.gate_activation, name="gated_candidate")
                memory_state = Add(memory_state, candidate, name="add_to_memory")

                # wrap memory transformation with something that can be treated as a layer
                memory_state = Module(inputs=[previous_memory,
                                              previous_h,
                                              input_layer],
                                      output=memory_state,
                                      name=self.name + "_memory")

            with tf.name_scope("output"):
                gate_o = Add(self.w_o, self.u_o, name="add_o")
                output = Activation(memory_state, fn=self.activation, name="output")
                current_h = Gate(output, gate_o, gate_fn=self.gate_activation, name="gated_output")

            if regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            current_h = Module(inputs=[input_layer, previous_h, previous_memory],
                               output=current_h,
                               name=self.name + "_h")

        return output.tensor, current_h, memory_state

    def reuse_with(self, input_layer, previous_state=None, regularized=None, name=None):
        return super().reuse_with(input_layer=input_layer,
                                  previous_state=previous_state,
                                  regularized=regularized,
                                  name=name,
                                  gate_activation=self.gate_activation,
                                  forget_bias_init=self.forget_bias_init)


class RNN(Layer):
    """ Recurrent Layer

    Takes a batch of sequences in time-major order [time_step,batch_size,feature_size]
    and dynamically unrolls a RecurrentCell applying it to each time step. The sequence
    should have at least one time step for which the recurrent cell is first created.
    After that, it supports an Unknown number of time steps. (time_step>=1)


    Args:
        input_seq: a Layer whose tensor has the shape [time_step,batch_size,feature_size] with time_step>=1
        cell_fn: a function that returns a cell when applied to a single timestep tensor of the form [batch_size,feature_size],
        the returned cell should have a ``regularized`` boolean parameter which appllies a regularized

    Attributes:
        cell: a Layer of type RecurrentCell used in the unrolled steps
        cell_proto (function): a function returning a RecurrentCell when applied to an input or tensor.
        This can be solved by creating a lambda with the sell parameters or a partial

    """

    def __init__(self,
                 input_seq,
                 cell_proto: Callable[[Union[Layer, tf.Tensor]], BaseRNNCell],
                 previous_state=None,
                 reverse=False,
                 regularized=False,
                 stateful=False,
                 share_vars_with: Optional['RNN'] = None,
                 name="rnn_layer"):

        self.cell_proto = cell_proto
        self.share_vars_with = share_vars_with
        self.cell = None
        self.regularized = regularized
        self.reverse = reverse
        self.previous_state = previous_state
        self.stateful = stateful

        # n_units and shape are set after the first cell is created
        super().__init__(input_layers=[input_seq] + as_list(previous_state),
                         n_units=None,
                         dtype=tf.float32,
                         name=name)

        tensor, state = self._build_graph()
        self.tensor = tensor
        self.state = [TensorLayer(s) for s in state]

    def _build_graph(self):
        input_seq = self.input_layers[0]

        with layer_scope(self):
            seq_len = tf.shape(input_seq)[0]
            input_ta = tf.TensorArray(dtype=input_seq.dtype, size=seq_len, tensor_array_name="inputs",
                                      clear_after_read=False)
            input_ta = input_ta.unstack(input_seq)
            output_ta = tf.TensorArray(dtype=self.dtype, size=seq_len, tensor_array_name="outputs")

            if self.reverse:
                i0 = seq_len - 1
                ii = i0 - 1
                fi = 0
            else:
                i0 = 0
                ii = i0 + 1
                fi = seq_len

            x0 = input_ta.read(i0)
            x0 = TensorLayer(x0)

            """
            Create CELL at t=1 or at the last time step if reverse
            """
            if self.share_vars_with is not None:
                cell = self.share_vars_with.cell
                cell = cell.reuse_with(input_layer=x0,
                                       previous_state=self.previous_state,
                                       regularized=self.regularized)
            else:
                cell = self.cell_proto(x0, previous_state=self.previous_state)
                if cell.regularized != self.regularized:
                    # create a new regularized cell if somehow the regularized parameter doesn't match the constructor
                    cell = cell.reuse_with(input_layer=x0,
                                           previous_state=self.previous_state,
                                           regularized=self.regularized)

            self.previous_state = cell.previous_state
            output_ta = output_ta.write(i0, cell.tensor)
            self.cell = cell
            self.n_units = cell.n_units

            def rnn_unroll(t, y, state):
                xt = input_ta.read(t)
                xt = TensorLayer(xt)
                c = cell.reuse_with(xt, previous_state=state)

                y = y.write(t, c.tensor)
                if self.reverse:
                    t = t - 1
                else:
                    t = t + 1
                return t, y, c.state

            i, out, last_state = tf.while_loop(cond=lambda t, *_: tf.math.not_equal(t, fi),
                                               body=rnn_unroll,
                                               loop_vars=(ii, output_ta, cell.state),
                                               name="rnn_unroll",
                                               parallel_iterations=1)

            # getting the results stores them in the previous state
            if self.stateful:
                updates = [zero_state.reuse_with(last_state, init_from_input=False).tensor
                           for zero_state, last_state in zip(cell.previous_state, last_state)]
                with tf.control_dependencies(updates):
                    out = out.stack()
            else:
                out = out.stack()
            return out, last_state

    def reuse_with(self, input_seq, previous_state=None, regularized=None, reverse=None, stateful=None, name=None):
        name = self.name if name is None else None
        regularized = self.regularized if regularized is None else regularized
        reverse = self.reverse if reverse is None else reverse
        share_vars_with = self.share_vars_with if self.share_vars_with is not None else self
        previous_state = self.previous_state if previous_state is None else previous_state
        stateful = self.stateful if stateful is None else stateful

        return RNN(input_seq=input_seq,
                   cell_proto=self.cell_proto,
                   regularized=regularized,
                   previous_state=previous_state,
                   stateful=stateful,
                   reverse=reverse,
                   share_vars_with=share_vars_with,
                   name=name)

    def reset(self):
        if self.stateful:
            return tf.group([state.reset() for state in self.cell.previous_state])
        else:
            return None


class SeqMap(Layer):
    """ Applies a given layer prototype to each element in the first dimension of the input layer

    """

    def __init__(self,
                 input_seq,
                 layer_proto: Callable[[Union[Layer, tf.Tensor]], Layer],
                 share_vars_with: Optional['SeqMap'] = None,
                 parallel_iterations=10,
                 name="seq_map"):

        self.layer_proto = layer_proto
        self.share_vars_with = share_vars_with
        self.layer_instance = None
        self.parallel_iterations = parallel_iterations

        # n_units and shape are set after the first cell is created
        super().__init__(input_layers=[input_seq],
                         n_units=None,
                         dtype=tf.float32,
                         name=name)

        tensor = self._build_graph()
        self.tensor = tensor

    def _build_graph(self):
        input_seq = self.input_layers[0]

        with layer_scope(self):
            seq_len = tf.shape(input_seq)[0]
            input_ta = tf.TensorArray(dtype=input_seq.dtype, size=seq_len, tensor_array_name="inputs",
                                      clear_after_read=False)
            input_ta = input_ta.unstack(input_seq)
            output_ta = tf.TensorArray(dtype=self.dtype, size=seq_len, tensor_array_name="outputs")

            i0 = 0
            ii = i0 + 1
            fi = seq_len

            x0 = input_ta.read(i0)
            x0 = TensorLayer(x0)

            """
            Create a layer instance at t=1
            """
            if self.share_vars_with is None:
                layer_instance = self.layer_proto(x0)
            else:
                layer_instance = self.share_vars_with.layer_instance
                layer_instance = layer_instance.reuse_with(x0)

            output_ta = output_ta.write(i0, layer_instance.tensor)
            self.layer_instance = layer_instance
            self.n_units = layer_instance.n_units

            def compute_step(t, y):
                xt = input_ta.read(t)
                xt = TensorLayer(xt)
                c = layer_instance.reuse_with(xt)
                y = y.write(t, c.tensor)
                t = t + 1
                return t, y

            i, out = tf.while_loop(cond=lambda t, *_: tf.math.not_equal(t, fi),
                                   body=compute_step,
                                   loop_vars=(ii, output_ta),
                                   name="map_seq",
                                   parallel_iterations=self.parallel_iterations)

            return out.stack()

    def reuse_with(self, input_seq, name=None):
        name = self.name if name is None else name

        return SeqMap(input_seq=input_seq,
                      layer_proto=self.layer_proto,
                      share_vars_with=self,
                      parallel_iterations=self.parallel_iterations,
                      name=name)


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
        lookup_shape: lookup table feature dimension
        batch_size: number of sequences to be looked up,
        if not None, will force a padding up to the specified batch_size
    """

    def __init__(self,
                 input_layer,
                 seq_size,
                 lookup_shape,
                 weight_init=glorot_uniform(),
                 batch_size=None,
                 bias=False,
                 shared_bias=None,
                 shared_weights=None,
                 dtype=tf.float32,
                 name="lookup",
                 share_vars_with=None,
                 batch_padding=True
                 ):

        input_layer = convert_to_layer(input_layer)

        self.weight_init = weight_init
        self.feature_shape = lookup_shape
        self.seq_size = seq_size
        self.batch_padding = batch_padding

        self.bias = bias
        self.shared_bias = shared_bias

        n_units = lookup_shape[-1]

        self.batch_size = batch_size

        self.share_vars_with = share_vars_with
        self.shared_weights = shared_weights

        if input_layer.is_sparse() and self.seq_size is None:
            raise ValueError("cannot use unknown seq_size with sparse inputs")

        if input_layer.is_dense() and input_layer.dtype not in (tf.int32, tf.int64):
            raise TypeError("invalid input layer dtype {}: should be {} or {}".format(
                input_layer.dtype,
                tf.int32,
                tf.int64
            ))

        if len(input_layer.shape) > 2:
            raise ValueError("expected 1D/2D input layer")
        elif input_layer.is_dense() and input_layer.n_units is not None:
            if seq_size is not None and input_layer.n_units > seq_size:
                raise ValueError("input layer n_units ({}) and seq_size ({}) should match for dense input layers \n"
                                 "if n_units < seq_size the lookup will be padded".format(input_layer.n_units,
                                                                                          seq_size))
        super().__init__(input_layers=input_layer, n_units=n_units, dtype=dtype, name=name)

        if self.shared_weights is not None:
            weight_shape = shared_weights.get_shape().as_list()
            if lookup_shape != weight_shape:
                raise ValueError(
                    "shared weight shape {} and feature shape {} mismatch".format(weight_shape, lookup_shape))

        if self.shared_bias is not None:
            num_bias = shared_bias.get_shape().as_list()[-1]
            if lookup_shape[0] != num_bias:
                raise ValueError(
                    "number of bias {} and number of feature rows {} mismatch".format(num_bias, lookup_shape[0]))

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

        self.tensor = self._build_graph()
        self.output_shape = [self.shape[0], self.seq_size, self.n_units]

    def _build_graph(self):
        input_layer = self.input_layers[0]
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            # with tf.name_scope(name) as scope, variable_scope.variable_scope(scope):
            # init weights

            if self.shared_weights is None:
                self.weights = tf.get_variable("weights",
                                               shape=self.feature_shape,
                                               initializer=self.weight_init,
                                               use_resource=True)
            else:
                self.weights = self.shared_weights

            self._add_variable(self.weights)

            if self.bias:
                if self.shared_bias is None:
                    self.bias = tf.get_variable("bias",
                                                shape=self.feature_shape[0],
                                                initializer=zeros_init(),
                                                use_resource=True)
                else:
                    self.bias = self.bias
            else:
                self.bias = None

            if self.bias is not None:
                self._add_variable(self.bias)

            # batch size is unknown for sparse lookups
            # y = xW
            if input_layer.is_sparse():

                input_tensor = input_layer.tensor
                sp_dim = tf.cast(input_tensor.dense_shape[-1], tf.int32)

                # transform.py 1D sparse lookups into 2D sparse lookup with 3 lookups
                # similar to the semantics of 1D dense tensor lookups
                if len(input_tensor.get_shape().as_list()) == 1:
                    sp_batch_size = tf.shape(input_tensor.values)[0]
                    sp_indices = transform.to_matrix_indices_2d(input_tensor.indices)
                    sp_batch_dim = tf.cast(tf.stack([sp_batch_size, sp_dim]), tf.int64)
                    input_tensor = tf.SparseTensor(sp_indices, input_tensor.values, sp_batch_dim)

                sp_values = input_tensor
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

                    lookup_bias = tf.expand_dims(lookup_bias, -1)

                    lookup_weights += lookup_bias

                tensor = lookup_weights

                # pad lookup if layer.tensor.dense_shape[0] is not a multiple of self.seq_size
                # this can happen if different lookups have a different number of indices
                lookup_batch = tf.shape(tensor)[0]
                expected_lookup_batch = tf.cast(
                    tf.math.ceil(lookup_batch / self.seq_size) * tf.cast(self.seq_size, dtype=tf.float64),
                    tf.int32)
                lookup_padding = expected_lookup_batch - lookup_batch

                # lookup_padding = sp_batch_size % self.seq_size
                lookup_padding = tf.stack([[0, lookup_padding], [0, 0]])
                tensor = tf.pad(tensor, lookup_padding)
                # tensor = Print(tensor,[tensor[0],tensor[1]],message="padded")

                # dynamic batch size with sparse tensors
                # batch_size = tf.cast(tf.math.ceil(sp_batch_size / self.seq_size), tf.int32)
                # batch_size = Print(batch_size, [batch_size], message="")
                # tensor = tf.reshape(tensor, tf.stack([-1, self.seq_size, self.n_units]))

                output_shape = tf.stack([-1, self.seq_size, self.n_units])
                tensor = tf.reshape(tensor, output_shape)

                # padding
                padding = []
                if self.batch_padding and self.batch_size is not None:
                    batch_padding = tf.math.maximum(self.batch_size - tf.shape(tensor)[0], 0)
                    padding.append([0, batch_padding])
                else:
                    padding.append([0, 0])

                padding.append([0, 0])
                padding.append([0, 0])

                padding = tf.stack(padding)
                tensor = tf.pad(tensor, padding)
            else:
                # layer is dense
                n_units = input_layer.n_units
                if n_units is None:
                    n_units = tf.shape(input_layer.tensor)[-1]

                # input_tensor = tf.reshape(input_layer.tensor, tf.stack([-1, n_units]))
                input_tensor = input_layer.tensor
                lookup_weights = tf.nn.embedding_lookup(params=self.weights,
                                                        ids=input_tensor)

                if self.bias is not None:
                    lookup_bias = tf.nn.embedding_lookup(params=self.bias,
                                                         ids=input_tensor)

                    lookup_bias = tf.expand_dims(lookup_bias, -1)
                    lookup_weights += lookup_bias

                batch_size = tf.shape(input_tensor)[0]
                lookup_shape = tf.stack([batch_size, -1, self.n_units])
                tensor = tf.reshape(lookup_weights, lookup_shape)

                # padding
                padding = []
                if self.batch_padding and self.batch_size is not None:
                    batch_padding = tf.math.maximum(self.batch_size - tf.shape(tensor)[0], 0)
                    padding.append([0, batch_padding])
                else:
                    padding.append([0, 0])

                # pad to seq_size if se_size is specified
                if self.seq_size is not None:
                    seq_padding = tf.math.maximum(self.seq_size - input_layer.n_units, 0)
                    padding.append([0, seq_padding])
                else:
                    padding.append([0, 0])

                padding.append([0, 0])
                padding = tf.stack(padding)
                tensor = tf.pad(tensor, padding)

        return tensor

    def as_concat(self):
        seq_size = self.seq_size
        if self.seq_size is None:
            seq_size = tf.shape(self.input_layers[-1].tensor)[-1]

        n_units = self.n_units * seq_size
        new_shape = tf.stack([-1, n_units])

        return WrapLayer(self,
                         n_units=n_units,
                         wrap_fn=lambda x: tf.reshape(x, new_shape),
                         attr_fwd=["weights", "bias", "seq_size"], name="concat")

    def permute_batch_time(self):
        return WrapLayer(layer=self,
                         n_units=self.n_units,
                         wrap_fn=lambda x: tf.transpose(x, [1, 0, 2]),
                         attr_fwd=["weights", "bias", "seq_size"], name="permute_batch_time")

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
        share_vars_with = self if self.share_vars_with is None else self.share_vars_with

        if name is None:
            name = self.name

        return Lookup(input_layer,
                      seq_size=self.seq_size,
                      lookup_shape=self.feature_shape,
                      batch_size=self.batch_size,
                      shared_weights=self.shared_weights,
                      weight_init=None,
                      dtype=self.dtype,
                      name=name,
                      share_vars_with=share_vars_with,
                      batch_padding=self.batch_padding)


def _apply_gate(layer: Layer, gate: Layer):
    with tf.name_scope("apply_gate", values=[layer.tensor, gate.tensor]):
        n_gates = tf.shape(gate.tensor)[-1]
        n_units = tf.shape(layer.tensor)[-1]

        feature_dim = n_units // n_gates

        if layer.is_sparse():
            tensor_in = tf.sparse_reshape(layer.tensor, [-1, n_gates, feature_dim])
            gated = mathx.sparse_multiply_dense(tensor_in, tf.expand_dims(gate.tensor, -1))
        else:
            tensor_in = tf.reshape(layer.tensor, [-1, n_gates, feature_dim])
            gated = tensor_in * tf.expand_dims(gate.tensor, -1)

        out_shape = tf.stack([-1, n_units])
        output = tf.reshape(gated, out_shape)

        # since n_units is taken from a tensor, we need to set the shape manually
        # otherwise this can't be determined
        output.set_shape([None, layer.n_units])

        return output


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
        # if layer.n_units is None:
        #     raise ValueError("n_units of layer to be gated cannot be None")
        #
        # if layer.tensor.get_shape()[-1] % gate_input.tensor.get_shape()[-1] != 0:
        #     raise ValueError("the n_units of the input layer {} is not a multiple of gate n_units {}".format(
        #         layer.n_units, gate_input.n_units))

        super().__init__(input_layers=[layer, gate_input],
                         n_units=layer.n_units,
                         dtype=tf.float32,
                         name=name)

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

        super().__init__(input_layers=[layer1, layer2, gate_input],
                         n_units=layer1.n_units,
                         dtype=tf.float32,
                         name=name)

        self.gate_fn = gate_fn
        self.gate_input = gate_input

        with layer_scope(self):
            self.gate1 = Activation(self.gate_input, self.gate_fn)
            self.gate2 = WrapLayer(self.gate1, n_units=self.gate1.n_units, wrap_fn=lambda x: 1 - x)

            output1 = _apply_gate(layer1, self.gate1)
            output2 = _apply_gate(layer2, self.gate2)
            output = tf.math.add(output1, output2)
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
        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=layer.name + "_sparse")

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
        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=layer.name + "_dense")

        with layer_scope(self):
            if layer.is_sparse():
                tensor = tf.sparse.to_dense(layer.tensor)
            else:
                tensor = layer.tensor
        self.tensor = tensor

    def reuse_with(self, layer):
        return ToDense(layer)


class ZoneOut(Layer):
    """ ZoneOut Layer.

    Zoneout is intended as a regularisation mechanism for recurrent neural networks to be applied between each
    unrolled step. The idea is to set the value of the current state of the neural network to its previous state
    if a unit is zoned-out.

    Reference:
       (Krueger et. al 2017) ZONEOUT: REGULARIZING RNNs BY RANDOMLY PRESERVING HIDDEN ACTIVATIONS
       https://arxiv.org/pdf/1606.01305.pdf

    Args:
        layer: an input layer :class:`Layer` to which ZoneOut will be applied to
        previous_layer: a layer to which ZoneOut will revert for each dropped unit in the input layer
        zoneout_prob: a scalar float with the probability that each element is dropped.
    """

    def __init__(self, layer, previous_layer, drop_prob=0.1, seed=None, mask=None, name="zoneout"):
        self.seed = seed
        self.drop_prob = drop_prob
        self.layer = layer
        self.previous_layer = previous_layer
        self.mask = mask

        if previous_layer.n_units != layer.n_units:
            raise ValueError("Can only apply zoneout to layers with the same n_units")

        n_units = layer.n_units
        super().__init__(input_layers=[layer, previous_layer],
                         n_units=n_units,
                         dtype=layer.dtype,
                         name=name)

        with layer_scope(self):
            if self.mask is None:
                mask_shape = tf.stack([tf.shape(layer.tensor)[0], self.n_units])
                mask = random_bernoulli(mask_shape, prob=1 - self.drop_prob, seed=seed)
                self.mask = mask

            # mask determines keep probability
            mask = TensorLayer(mask, n_units=self.n_units, dtype=layer.dtype)
            gate = CoupledGate(layer1=layer,
                               layer2=previous_layer,
                               gate_input=mask,
                               gate_fn=identity,
                               name="zoneout_gate")
        self.tensor = gate.tensor

    def reuse_with(self, current_layer, previous_layer, name=None, share_mask=True):
        if name is None:
            name = self.name

        mask = self.mask if share_mask else None
        return ZoneOut(current_layer,
                       previous_layer,
                       drop_prob=self.drop_prob,
                       seed=self.seed,
                       mask=mask,
                       name=name)


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
                         dtype=layer.dtype,
                         name=layer.name + "_gaussian_noise")

        with layer_scope(self):
            if layer.is_sparse():
                tensor = tf.sparse.to_dense(layer.tensor)
            else:
                tensor = layer.tensor

            noise_shape = tf.shape(tensor)
            noise = tf.random_normal(noise_shape, mean, stddev, seed=seed, dtype=tf.float32)

            tensor = tf.cast(tensor, tf.float32)
            tensor = tf.math.add(tensor, noise)

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

        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=layer.name + "_sp_noise")

        # do nothing if amount of noise is 0
        if density == 0.0 or self.num_corrupted() == 0:
            tensor = layer.tensor
        else:
            with layer_scope(self):

                batch_size = tf.shape(layer.tensor, out_type=tf.int64)[0]

                noise = salt_pepper_noise(dim=self.n_units,
                                          batch_size=batch_size,
                                          density=density,
                                          salt_value=salt_value,
                                          pepper_value=pepper_value,
                                          seed=seed)

                if layer.is_sparse():
                    tensor = transform.sparse_put(layer.tensor, noise)
                else:
                    tensor = transform.dense_put(layer.tensor, noise)

        self.tensor = tensor

    def num_corrupted(self):
        """ Returns the number of entries corrupted by noise per sample"""
        num_noise = int(self.density * self.n_units)

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

    def __init__(self, layer, fn=tf.identity, name="activation", **keywords):
        self.fn = partial(fn, **keywords)
        self.kw = keywords
        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=name)

        with layer_scope(self):
            if layer.is_sparse():
                tensor = tf.sparse.to_dense(layer.tensor)
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

        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=bias_name)

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
            self.bias = tf.get_variable("bias",
                                        shape=[self.n_units],
                                        initializer=zeros_init(),
                                        use_resource=True)
            self._add_variable(self.bias)
            if layer.is_sparse():
                tensor = tf.sparse.to_dense(layer.tensor)
            else:
                tensor = layer.tensor
                tensor = tf.nn.bias_add(tensor, self.bias, name="tensor")

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
                 merge_fn=tf.math.add_n,
                 name="merge"):

        self.weights = weights
        self.merge_fn = merge_fn

        if len(layers) < 1:
            raise Exception("You must provide at least one layer")

        if weights is not None and len(weights) != len(layers):
            raise Exception("len(weights) must be equals to len(layers)")

        with layer_scope(self, name=name):
            layers = list(map(lambda l: TensorLayer(l) if not isinstance(l, Layer) else l, layers))
            if weights is not None:
                tensors = [tf.math.scalar_mul(weights[i], layers[i].tensor) for i in range(len(layers))]
            else:
                tensors = [layer.tensor for layer in layers]
            tensor = merge_fn(tensors)

        output_shape = tensor.get_shape()
        output_shape = output_shape.as_list()
        self.output_shape = output_shape

        shape = [[layer.n_units for layer in layers], output_shape[-1]]
        n_units = output_shape[-1]

        super().__init__(input_layers=layers, n_units=n_units, dtype=tensor.dtype, name=name)

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
        layers = list(map(lambda l: TensorLayer(l) if not isinstance(l, Layer) else l, layers))

        def merge_add(tensors):
            res = 0
            for tensor in tensors:
                res = res + tensor
            return res

        super().__init__(*layers, weights=weights, merge_fn=merge_add, name=name)


class Mean(Merge):
    """ Merges the outputs of multiple layers with the same shape by computing their mean value
    """

    def __init__(self, *layers, weights=None, name="mean"):
        super().__init__(*layers, weights=weights, merge_fn=partial(tf.math.reduce_mean, axis=0), name=name)


class SeqConcat(Layer):
    """ Concat 3D Layer representing a sequence of vectors

    """

    def __init__(self, input_layer, time_major=True, seq_size=None, name="seq_concat"):
        super().__init__(input_layers=input_layer, n_units=None, name=name)
        with layer_scope(self):
            n_units = input_layer.n_units
            if time_major:
                input_layer = Transpose(input_layer, [1, 0, 2])
                dynamic_seq_size = tf.shape(input_layer)[0]
            else:
                dynamic_seq_size = tf.shape(input_layer)[1]

            if seq_size is None:
                seq_size = dynamic_seq_size

            new_n_units = tf.shape(input_layer.tensor)[-1] * seq_size

            static_seq_size = tx_utils.static_value(dynamic_seq_size)
            # if this can't be computed we can't use the static value
            if seq_size is not None and static_seq_size is not None and seq_size != static_seq_size:
                raise ValueError("seq_size and number of units mismatch {}!={}".format(seq_size, static_seq_size))

            if static_seq_size is not None:
                self.n_units = static_seq_size
            elif seq_size is not None:
                self.n_units = n_units * seq_size

            self.tensor = tf.reshape(input_layer, [-1, new_n_units])


class Concat(Layer):
    """ Concat Layer

    Concatenates input layers on the last dimension

    Args:
        layers: a :obj:`list` of :class:`Layer`
        name: name for the layer scope
    """

    def __init__(self, *layers, name="concat"):
        layers = list(map(lambda l: TensorLayer(l) if not isinstance(l, Layer) else l, layers))
        first, *rest = layers
        if not all(layer.dtype == first.dtype for layer in rest):
            raise ValueError("Layers must have the same type to be concatenated")

        total_units = sum([layer.n_units for layer in layers])
        super().__init__(input_layers=layers, n_units=total_units, dtype=first.dtype, name=name)

        with layer_scope(self):
            tensors = [layer.tensor for layer in layers]
            tensor = tf.concat(tensors, axis=-1)

        self.tensor = tensor

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name
        return Concat(*layers, name)


class Highway(Layer):
    def __init__(self, x_layer, h_layer,
                 transform_weight_init=glorot_uniform(),
                 transform_bias_init=const_init(-2),
                 carry_gate=False,
                 carry_weight_init=glorot_uniform(),
                 carry_bias_init=zeros_init(),
                 share_vars_with=None,
                 name="highway"):

        self.transform_weight_init = transform_weight_init
        self.transform_bias_init = transform_bias_init
        self.carry_weight_init = carry_weight_init
        self.carry_bias_init = carry_bias_init
        self.carry_gate = carry_gate

        self.share_vars_with = share_vars_with

        # try to create a module from the x_layer -> h_layer
        # if one is not connected to the other, this fails
        self.module = Module(x_layer, h_layer)

        if share_vars_with is not None:
            if not isinstance(share_vars_with, Highway):
                raise TypeError("can only share vars with a Highway Layer {} found".format(type(share_vars_with)))

        if x_layer.n_units != h_layer.n_units:
            raise ValueError("The input x_layer should have the same n_units as the h_layer {}!={}".format(
                x_layer.shape, h_layer.shape
            ))

        super().__init__(input_layers=[x_layer, h_layer],
                         n_units=x_layer.n_units,
                         dtype=x_layer.dtype,
                         name=name)

        with layer_scope(self):
            if self.share_vars_with is None:
                self.t_gate = Linear(x_layer, x_layer.n_units,
                                     weight_init=self.transform_weight_init,
                                     bias_init=self.transform_bias_init,
                                     name="w_t")
                if self.carry_gate:
                    self.c_gate = Linear(x_layer, x_layer.n_units,
                                         weight_init=self.carry_weight_init,
                                         bias_init=self.carry_bias_init,
                                         name="w_c")
            else:
                self.t_gate = self.share_vars_with.t_gate.reuse_with(x_layer)
                if self.carry_gate:
                    self.c_gate = self.share_vars_with.c_gate.reuse_with(x_layer)

            if not self.carry_gate:
                output = CoupledGate(h_layer, x_layer, gate_input=self.t_gate)
            else:
                gated_x = Gate(x_layer, gate_input=self.c_gate)
                gated_h = Gate(h_layer, gate_input=self.t_gate)
                output = Add(gated_h, gated_x)

        self.tensor = output.tensor

    def reuse_with(self, x_layer, h_layer, name=None):
        if name is None:
            name = self.name

        share_vars_with = self if self.share_vars_with is None else self.share_vars_with

        return Highway(x_layer=x_layer,
                       h_layer=h_layer,
                       transform_weight_init=self.transform_weight_init,
                       transform_bias_init=self.transform_bias_init,
                       carry_gate=self.carry_gate,
                       carry_weight_init=self.carry_weight_init,
                       carry_bias_init=self.carry_bias_init,
                       share_vars_with=share_vars_with,
                       name=name)


class Residual(Layer):
    """ Residual Block



    """

    def __init__(self, x_layer, h_layer, share_vars_with=None, weight_init=glorot_uniform(), name="residual"):

        # try to create a module from the x_layer -> h_layer
        # if one is not connected to the other, this fails
        self.module = Module(x_layer, h_layer)
        self.weight_init = weight_init
        self.share_vars_with = share_vars_with
        self.projection = x_layer

        if share_vars_with is not None:
            if not isinstance(share_vars_with, Residual):
                raise TypeError("can only share vars with a Residual Layer {} found".format(type(share_vars_with)))

        super().__init__(input_layers=[x_layer, h_layer],
                         n_units=h_layer.n_units,
                         dtype=h_layer.dtype,
                         name=name)

        with layer_scope(self):
            # we need to perform a linear projection so that the dimensions match
            if x_layer.n_units != h_layer.n_units:
                if self.share_vars_with is None:
                    self.projection = Linear(x_layer, h_layer.n_units, weight_init=weight_init, add_bias=False)
                else:
                    self.projection = share_vars_with.projection.reuse_with(x_layer)

                self._add_variable(self.projection.weights)

            output = Add(h_layer, self.projection)

        self.tensor = output.tensor

    def reuse_with(self, x_layer, h_layer, name=None):
        if name is None:
            name = self.name

        share_vars_with = self if self.share_vars_with is None else self.share_vars_with

        return Residual(x_layer=x_layer,
                        h_layer=h_layer,
                        share_vars_with=share_vars_with,
                        name=name)


class Reshape(Layer):
    def __init__(self, layer, shape, name="reshape"):
        if not isinstance(layer, Layer):
            layer = TensorLayer(layer)
        self.target_shape = [d if d is not None else -1 for d in shape]

        with layer_scope(self, name=name):
            if layer.is_dense():
                tensor = tf.reshape(layer.tensor, self.target_shape)
            else:
                tensor = tf.sparse_reshape(layer.tensor, self.target_shape)

        n_units = tensor.get_shape().as_list()[-1]
        shape = [layer.n_units, n_units]

        super().__init__(input_layers=layer, n_units=n_units, dtype=layer.dtype, name=name)

        self.tensor = tensor

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name
        return Reshape(layer, self.target_shape, name)


class Transpose(Layer):
    def __init__(self, layer, perm=None, name="transpose"):
        if not isinstance(layer, Layer):
            layer = TensorLayer(layer)
        self.perm = perm
        with layer_scope(self, name=name):
            if layer.is_dense():
                output = tf.transpose(layer.tensor, perm)
            else:
                output = tf.sparse.transpose(layer.tensor, perm)

        n_units = output.get_shape().as_list()[-1]

        super().__init__(input_layers=layer, n_units=n_units, dtype=layer.dtype, name=name)
        self.tensor = output

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name
        return Transpose(layer, self.perm, name)


class Flatten(Layer):
    """ Flatten.

    Flattens the input layer without changing the batch-size

    """

    def __init__(self, layer, name="flatten"):
        if not isinstance(layer, Layer):
            layer = TensorLayer(layer)
        n_units = reduce(operator.mul, layer.tensor.get_shape().as_list()[1:])
        super().__init__(input_layers=layer, n_units=n_units, dtype=layer.dtype, name=name)

        with layer_scope(self):
            input_shape = tf.shape(layer.tensor)

            output = layer.tensor
            if layer.is_dense():
                output = tf.reshape(output, tf.stack([-1, tf.math.reduce_prod(input_shape[1:])]))
            else:
                output = tf.sparse_reshape(output, tf.stack([-1, tf.math.reduce_prod(input_shape[1:])]))

        self.tensor = output


class BatchNorm(Layer):
    """ Batch Normalization Layer

    Usage:
        Typically, what we want to do is setup the inference graph or training graph first with all the BatchNorm
        layers in place, afterwards we can call reuse_with with a different value on the training flag. This will
        create the appropriate batch_norm computations for training and inference time while sharing the same variables.

        BatchNorm is better understood as a technique which reduces second-order relationships between parameters of
        different layers than a method to reduce covariate shift. Thus, the before/after distinction doesn't
        matter, and differences in performance could simply be because of other particular factors of the model.


        Training time:
            * computes batch normalisation based on mini-batch mean and variance
            * computes the population estimates based on a exponential moving average with a given decay parameter

        Inference time:
            * computes batch normalisation based on population estimates for mean and variance (learned during training)


        How to use center and scale params:
            * if you use center=True, your preceeding layer does not require a bias because this bias will be canceled
            out in the batch_norm process anyway.

            * when the next layer is linear (e.g. a ReLU Activation), scale can be set to False, since the scaling can
            be done by the next layer if needed.

    Impl Note:
        if I added the updates to the update collection I would have to do something like

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        Don't see the need for now, this seems ugly. Perhaps later I can add this info to any layer (updates
        that need to take place)

    References:

        Ioffe, Szegedy "Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift"

        from http://arxiv.org/abs/1502.03167.

    Args:
        layer: Layer from which the batch normalisation will be computed
        training: if True uses the current mini-batch mean and variance to compute the batch-normalised output and
        updates the population estimates. Else, computes the batch normalisation using the estimates.
        center: If True, subtract `beta`. If False, `beta` is ignored.
        scale: If True, multiply by `gamma`. If False, `gamma` is not used.


    """

    def __init__(self,
                 layer,
                 training=True,
                 center=True,
                 scale=False,
                 trainable=True,
                 gamma_init=ones_init(),
                 beta_init=zeros_init(),
                 decay=0.99,
                 epsilon=0.001,
                 gamma=None,
                 beta=None,
                 moving_mean=None,
                 moving_variance=None,
                 share_vars_with=None,
                 name="BatchNorm"):

        self.decay = decay
        self.epsilon = epsilon
        self.training = training
        self.center = center
        self.beta = beta
        self.scale = scale
        self.gamma = gamma
        self.trainable = trainable
        self.share_vars_with = share_vars_with
        self.gamma_init = gamma_init
        self.beta_init = beta_init
        self.moving_mean = moving_mean
        self.moving_variance = moving_variance

        input_shape = layer.tensor.get_shape().as_list()
        param_shape = input_shape[-1:]
        self.param_shape = param_shape
        dtype = layer.dtype

        # validate beta and gamma, and possible shared vars
        _validate_shape_type(gamma, param_shape)
        _validate_shape_type(beta, param_shape)
        _validate_shape_type(moving_mean, param_shape, dtype)
        _validate_shape_type(moving_variance, param_shape, dtype)

        if layer.dtype not in (tf.float32, tf.float64, tf.float16):
            raise TypeError("Expected float layer got {} instead".format(layer.dtype))

        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=name)
        self.tensor = self._build_graph()

    def reset_estimates(self):
        reset_mean = tf.assign(self.moving_mean, tf.zeros_like(self.moving_mean))
        reset_variance = tf.assign(self.moving_variance, tf.zeros_like(self.moving_variance))

        return tf.group(reset_mean, reset_variance)

    def _build_graph(self):
        input_layer = self.input_layers[0]
        var_scope_name = self.share_vars_with.scoped_name if self.share_vars_with is not None else None
        var_reuse = self.share_vars_with is not None

        with layer_scope(self, var_scope=True, var_reuse=var_reuse, var_scope_name=var_scope_name):
            if input_layer.is_sparse():
                input_layer = ToDense(input_layer)

            input_shape = input_layer.tensor.get_shape().as_list()
            axis = list(range(len(input_shape) - 1))

            if self.scale:
                if self.gamma is None:
                    self.gamma = tf.get_variable("gamma",
                                                 shape=self.param_shape,
                                                 dtype=self.dtype,
                                                 initializer=self.gamma_init,
                                                 trainable=self.trainable,
                                                 use_resource=True)

                # store variables for easy access
                self._add_variable(self.gamma)
            else:
                self.gamma = None
            if self.center:
                if self.beta is None:
                    self.beta = tf.get_variable("beta",
                                                shape=self.param_shape,
                                                dtype=self.dtype,
                                                initializer=self.beta_init,
                                                trainable=self.trainable,
                                                use_resource=True)

                # store variables for easy access
                self._add_variable(self.beta)
            else:
                self.beta = None

            if self.moving_mean is None:
                self.moving_mean = tf.get_variable("moving_mean",
                                                   shape=self.param_shape,
                                                   initializer=zeros_init(),
                                                   trainable=False,
                                                   use_resource=True,
                                                   dtype=self.dtype)
            self._add_variable(self.moving_mean)

            if self.moving_variance is None:
                self.moving_variance = tf.get_variable("moving_variance",
                                                       shape=self.param_shape,
                                                       initializer=zeros_init(),
                                                       trainable=False,
                                                       use_resource=True,
                                                       dtype=self.dtype)

            self._add_variable(self.moving_variance)

            # Calculate the moments based on the individual batch.
            batch_mean, batch_variance = tf.nn.moments(input_layer.tensor, axis, shift=self.moving_mean, name="moments")

            # self.moments = batch_mean

            # I have to create this graph regardless of weather I'm training or not because
            # of variable sharing, inside this op, there's an attempt to create a variable
            # with a name based on the self.moving_mean_op name BatchNorm/moving_mean
            # if we are sharing variables this is already scoped, the new variable will
            # repeat the scope BatchNorm/BatchNorm/moving_mean/biased

            # zero de-bias ema update
            with tf.variable_scope("debias"):
                update_mv_avg = moving_averages.assign_moving_average(self.moving_mean, batch_mean, self.decay,
                                                                      zero_debias=True)
                update_mv_var = moving_averages.assign_moving_average(self.moving_variance, batch_variance, self.decay,
                                                                      zero_debias=True)

        if self.training:
            with tf.control_dependencies([update_mv_avg, update_mv_var]):
                return tf.nn.batch_normalization(x=input_layer.tensor,
                                                 mean=batch_mean,
                                                 variance=batch_variance,
                                                 offset=self.beta,
                                                 scale=self.gamma,
                                                 variance_epsilon=self.epsilon)
        else:
            return tf.nn.batch_normalization(x=input_layer.tensor,
                                             mean=self.moving_mean,
                                             variance=self.moving_variance,
                                             offset=self.beta,
                                             scale=self.gamma,
                                             variance_epsilon=self.epsilon)

    def reuse_with(self, layer, training=None, name=None):
        if training is None:
            training = self.training

        if name is None:
            name = self.name

        share_vars_with = self if self.share_vars_with is None else self.share_vars_with

        return BatchNorm(layer=layer,
                         training=training,
                         center=self.center,
                         scale=self.scale,
                         trainable=self.trainable,
                         gamma=self.gamma,
                         beta=self.beta,
                         gamma_init=self.gamma_init,
                         beta_init=self.beta_init,
                         decay=self.decay,
                         epsilon=self.epsilon,
                         share_vars_with=share_vars_with,
                         name=name)


class LayerNorm(Layer):
    """ Layer Normalization

    References:
        (Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, 2016) "Layer Normalization"
        https://arxiv.org/abs/1607.06450.

    """

    def __init__(self, input_layer, share_state_with=None):
        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=tf.float32,
                         name="layer_normalization"
                         )

        self.share_state_with = share_state_with
        self.tensor = self._build_graph()

    def _build_graph(self):
        input_layer = self.input_layers[0]

        with layer_scope(self):
            if self.share_state_with is None:
                # center
                self.beta = tf.get_variable("beta", shape=[input_layer.n_units], initializer=zeros_init())
                # scale
                self.gamma = tf.get_variable("gamma", shape=[input_layer.n_units], initializer=ones_init())
            else:
                self.beta = self.share_state_with.beta
                self.gamma = self.share_state_with.gamma

            mean, variance = tf.nn.moments(input_layer.tensor, -1, keep_dims=True)
            variance_epsilon = 1e-12

            return tf.nn.batch_normalization(
                input_layer.tensor,
                mean,
                variance,
                offset=self.beta,
                scale=self.gamma,
                variance_epsilon=variance_epsilon)

    def reuse_with(self, input_layer, name=None):
        return LayerNorm(input_layer, share_state_with=self)


class Attention(Layer):
    """ Scaled Dot Product MultiHead Attention Layer

    Args:
        input_layer
        n_units: output number of units, each attentio head has n_units // n_head units, meaning that n_units needs to
        be a multiple of n_heads.

    """

    def __init__(self,
                 query_layer,
                 key_layer,
                 value_layer,
                 n_units=None,
                 n_heads=1,
                 attention_fn=tf.nn.softmax,
                 causality=False,
                 attention_dropout=0.0,
                 regularized=False,
                 name="attention",
                 share_state_with=None):
        self.n_heads = n_heads
        n_units = query_layer.n_units if n_units is None else n_units
        self.causality = causality
        self.share_shate_with = share_state_with
        self.regularized = regularized
        self.attention_dropout = attention_dropout
        self.attention_fn = attention_fn

        if n_units % n_heads != 0:
            raise ValueError(
                "The n_units {} is not a multiple of the number of attention "
                "heads {}".format(self.n_units, n_heads))

        self.head_units = n_units // n_heads
        super().__init__(input_layers=[query_layer, key_layer, value_layer], n_units=n_units, name=name)
        self.tensor = self._build_graph()

    def _build_graph(self):
        q, k, v = self.input_layers
        h_dim = self.n_units
        state = self.share_shate_with
        # input_dim = input_layer.tensor.get_shape().as_list()[-1]

        with layer_scope(self):
            if state is None:
                # (batch_size, t, n_units)
                self.wq = Linear(q, n_units=h_dim, add_bias=False, name="wq")
                self.wk = Linear(k, n_units=h_dim, add_bias=False, name="wk")
                self.wv = Linear(v, n_units=h_dim, add_bias=False, name="wv")
                self.bv = Bias(self.wv, name="bv")
            else:
                self.wq = state.wq.reuse_with(q)
                self.wk = state.wq.reuse_with(k)
                self.wv = state.wq.reuse_with(v)
                self.bv = state.wq.reuse_with(self.wv)

            # (n_heads*batch_size, steps, n_units//n_heads)
            qh = tf.concat(tf.split(self.wq, self.n_heads, axis=2), axis=0)
            kh = tf.concat(tf.split(self.wk, self.n_heads, axis=2), axis=0)
            vh = tf.concat(tf.split(self.wv, self.n_heads, axis=2), axis=0)

            dotprod = tf.matmul(qh, tf.transpose(kh, [0, 2, 1]))
            dotprod /= h_dim ** 0.5
            output = dotprod

            # mask information from the future
            if self.causality:
                diag_values = tf.ones_like(output[0, :, :])  # (tq, tk)
                triangular = tf.linalg.LinearOperatorLowerTriangular(diag_values).to_dense()  # (tq, tk)
                masks = tf.tile(tf.expand_dims(triangular, 0), [tf.shape(output)[0], 1, 1])  # (N, tq, tk)
                # mask to - inf before softmax
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                output = tf.where(tf.equal(masks, 0), paddings, output)

            attention_scores = self.attention_fn(output)

            if self.attention_dropout > 0 and self.regularized:
                attention_scores = Dropout(attention_scores,
                                           probability=self.attention_dropout,
                                           scale=True,
                                           locked=False,
                                           name="attention_dropout")

            # weighted sum (context vectors) weighted by attention scores
            context_vectors = tf.matmul(attention_scores, vh)
            # restore shape (batch_size, tq, n_units)
            output = tf.concat(tf.split(context_vectors, self.n_heads, axis=0), axis=2)

        return output

    def reuse_with(self, query_layer, key_layer, value_layer, regularized=None, name=None):
        regularized = self.regularized if regularized is None else regularized
        name = self.name if name is None else name

        return Attention(query_layer=query_layer,
                         key_layer=key_layer,
                         value_layer=value_layer,
                         n_units=self.n_units,
                         attention_fn=self.attention_fn,
                         n_heads=self.n_heads,
                         attention_dropout=self.attention_dropout,
                         regularized=regularized,
                         name=name,
                         share_state_with=self)


def convert_to_layer(layer_or_tensor: Union[tf.Tensor, Layer], dtype=None):
    """ Converts tensor or tensor convertible object to Layer

    Args:
        dtype: type for our converted layer
        layer_or_tensor: a tensor or convertible to tensor (tx.utils.to_tensor_cast(tensor))

    Returns:
        a TensorLayer wrapping the given tensor
    """
    if isinstance(layer_or_tensor, Layer):
        return layer_or_tensor
    else:
        tensor = tx_utils.to_tensor_cast(layer_or_tensor, dtype)
        return TensorLayer(tensor)


class Param(Layer):
    """ Param

    a special building block to pass scalar parameter to neural network models with support for default values

    Args:
        value: initial Parameter value, if None must be fed

    """

    def __init__(self, value=None, dtype=tf.float32, name="param"):
        super().__init__(input_layers=[], n_units=0, dtype=dtype, name=name)
        self.property = Property(name=self.name, value=value)

        with layer_scope(self, name=name):
            self.placeholder = tf.placeholder(dtype=self.dtype, shape=[], name=self.name)
            self.tensor = self.placeholder

    # forward methods to property including registry
    @property
    def value(self):
        return self.property.value

    @value.setter
    def value(self, value):
        self.property.value = value

    def register(self, obs):
        self.property.register(obs)

    def reuse_with(self, *layers, name=None):
        raise AttributeError("Cannot call reuse_with on Param")

    def __call__(self, *args, **kwargs):
        return self.tensor

    def eval(self, feed_dict=None, session=None):
        if feed_dict is not None:
            if self.placeholder not in feed_dict:
                raise ValueError("feed dict does not contain the placeholder in this Param")
        elif self.value is not None:
            feed_dict = {self.placeholder: self.value}
        return self.tensor.eval(feed_dict, session)

    def __str__(self):
        return "{name}::{cname}({dtype})".format(name=self.name, cname=type(self).__name__, dtype=self.dtype)


# register Layer as Tensor
def layer_to_tensor(layer, dtype=None, name=None, as_ref=False):
    with tf.name_scope(name):
        return tx_utils.to_tensor_cast(layer.tensor, dtype=dtype)


tf.register_tensor_conversion_function(
    base_type=Layer,
    conversion_func=layer_to_tensor,
    priority=100
)

__all__ = ["Input",
           "FC",
           "RNNCell",
           "GRUCell",
           "LSTMCell",
           "RNN",
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
           "WrapLayer",
           "Module",
           "ZoneOut",
           "Conv1D",
           "CausalConv",
           "QRNN",
           "Highway",
           "Residual",
           "Flatten",
           "Reshape",
           "Transpose",
           "BatchNorm",
           "Conv2D",
           "VariableLayer",
           "LambdaLayer",
           "Mean",
           "DropConnect",
           "ViewLayer",
           "Param",
           "SeqMap",
           "Attention",
           "SeqConcat",
           "LayerNorm",
           "DropLookup"
           ]
