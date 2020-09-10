from abc import ABC
import tensorflow as tf
from tensorx.utils import as_tensor, as_list, Graph
from typing import Union, Type, Callable, Optional, List
import inspect
from contextlib import ExitStack
import tensorx.ops as txf
from tensorx.train.callbacks import OnValueChange
from functools import partial
from tensorx.ops import embedding_lookup_sparse
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.training.tracking.base import TrackableReference
from tensorflow.python.training.tracking import data_structures as track

import tensorx as tx
import threading
from collections import Counter
import re


class LayerState(AutoTrackable):
    """ Layer state is used as a namespace to store either ``tf.Variable`` or ``Layer`` instances
    that contain ``tf.Variables`` that define the state of a ``Layer``.

    !!! note ""
        ``LayerState`` objects contain both `tf.Variable` and `Layer` objects.
    """

    def __init__(self):
        pass
        # don't repeat tracking on layer
        # self.layer = track.NoDependency(origin_layer)

    def variables(self):
        """ returns a list of all variables contained in the layer state object

        Returns:
            list: a list of all variables in the layer state

        """
        return list(self.var_dict().values())

    def var_dict(self):
        """ returns a dictionary where the keys are the var refs

        Returns:
            dict: a dictionary from attribute names to variable instances with all
            the variables in the current layer state

        """
        all_vars = dict()
        for attr, obj in self.__dict__.items():
            # TODO added this to test layer states that have a None
            # if obj is None:
            #    raise ValueError(f"object for {attr} has None value")
            if isinstance(obj, Layer):  # and not obj == self.layer:
                v = obj.layer_state.var_dict()
                all_vars.update(v)
            elif isinstance(obj, tf.Variable):
                # Tensors are not hashable but this gets the object id
                all_vars[obj.ref()] = obj
                # all_vars[attr] = obj
        return all_vars

    def __str__(self):
        return "State{\n %s \n}" % ("\n".join(["\t{}: {}".format(k, str(i)) for k, i in self.__dict__.items()]))


class LayerScope:
    """ LayerScope creates a unique name for the scope if not executing eagerly

    Args:
        current_layer: layer to be used in this scope, the layer name is used as scope name for tensorflow tf.name_scope
    """

    def __init__(self, current_layer, name=None):
        self.layer = current_layer
        if name is not None and current_layer is not None:
            self.layer.name = name

        if current_layer is None and name is None:
            raise ValueError("layer scope needs a Layer or a name but both are None")

        self.name = self.layer.name if current_layer is not None else name
        self._stack = None

    def __enter__(self):
        with ExitStack() as stack:
            layer_name_scope = tf.name_scope(self.name)
            scoped_name = stack.enter_context(layer_name_scope)
            scoped_name = scoped_name[:-1]

            self._stack = stack.pop_all()
            return scoped_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)


layer_scope: Type[LayerScope] = LayerScope


# TODO order arguments while accepting kwargs IF the classes in question allow kwargs in the constructors
#  check if the args order themselves in case I present something that does is not named
class LayerProto:
    """ Layer Proto

    Creates a Layer prototype. A callable that allows us to delay calling the constructor a Layer
    object and validates constructor arguments.

    """

    def __init__(self, layer_cls, **kwargs):
        self.layer_cls = layer_cls
        self.arg_spec = inspect.getfullargspec(layer_cls.__init__)
        self.arg_names = set(self.arg_spec.args[1:] + self.arg_spec.kwonlyargs)
        self._validate_args(**kwargs)
        self.arg_dict = kwargs

    def filter_args(self, **kwargs):
        new_kwargs = dict(kwargs)
        for key in kwargs:
            if key not in self.arg_names and not self.arg_spec.varkw:
                del new_kwargs[key]
        return new_kwargs

    def _validate_args(self, **kwargs):
        for key in kwargs:
            if key not in self.arg_names and not self.arg_spec.varkw:
                raise TypeError(f"{self.layer_cls.__name__} prototype got an unexpected argument {key}\n")

    def __repr__(self):
        kw = ', '.join([f'{k}={repr(v)}' for k, v in self.arg_dict.items()])
        return f"{type(self).__name__}({self.layer_cls.__name__},{kw})"

    def __call__(self, *args, **kwargs):
        new_args = dict(self.arg_dict)
        new_args.update(kwargs)
        return self.layer_cls(*args, **new_args)

    def update(self, **kwargs):
        """ update.

        Updates the prototype constructor argument dictionary and validates those parameters.

        Args:
            **kwargs: new values for constructor named arguments to be updated
        """
        self._validate_args(**kwargs)
        self.arg_dict.update(kwargs)


class Layer(AutoTrackable):
    """ Layer base class

    Attributes:
        input_layers: a list of input nodes for the current layer
        n_units: the number of units for the current layer (last dim)
        name: name to be used for the layer scope
        scoped_name: layer full scope name

    Notes:
        * If a layer is created inside the scope of another layer, its scoped_name is the final name attributed
        to the layer taking the outer scopes into account.

    Args:
        input_layers: a single layer,a list of input layers, or None if no inputs are required
        n_units: dimension of input vector (dimension of columns in case batch_size != None
        name: layer name (used to nam the placeholder)

    """
    NAMES = Counter()
    NAME_LOCK = threading.RLock()

    def __init__(self, input_layers, n_units, dtype=None, name="layer", **kwargs):
        self.n_units = n_units

        with Layer.NAME_LOCK:
            if name in Layer.NAMES:
                Layer.NAMES[name] += 1
                self.name = name + f"_{Layer.NAMES[name]}"
            else:
                Layer.NAMES[name] = 1
                self.name = name
        with tf.name_scope(self.name) as scope:
            # scoped_name = stack.enter_context(layer_name_scope)
            # scoped_name = layer_name_scope
            self.scoped_name = scope[:-1]

        # self.scoped_name = name

        self.dtype = tf.dtypes.as_dtype(dtype) if dtype is not None else None

        # a cleaner way to set attributes is to add kwargs to the super().__init__(...attr=value)
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # proto is built from all __dict__ and constructor argspec
        proto = type(self).proto()
        new_args = proto.filter_args(**self.__dict__)
        proto.update(**new_args)
        self.proto = proto
        self._input_layers = [as_layer(input_layer) for input_layer in as_list(input_layers)]

        self.layer_state = self.init_state()

        # TODO I think referring to state explicitly should be the preferred way to
        #   access it but this is only a problem if a shared state changes somehow
        #   e.g. two linear layers one has bias the second does not, if we add a bias
        #       to one that shares a state, it has to reuse the bias from the second layer
        # forward attributes from state to avoid layer.layer_state.variable
        if self.layer_state is not None:
            self.__dict__.update(self.layer_state.__dict__)

        # save this for __call__
        # TODO this should be built lazily for __call__ instead of storing a subgraph on each node
        if self.input_layers:
            input_graph: Graph = Graph.build(inputs=None,
                                             outputs=self.input_layers)
            self.input_graph: Graph = track.NoDependency(input_graph)

    def init_state(self):
        self.layer_state = LayerState()
        return self.layer_state

    def __setattr__(self, key, value):
        """ if key is in layer_state, changes layer state as well
        Note:
            During debugging I noticed that changing an instance member shared with the layer state would not
            change the layer state, this could be problematic if users are trying to debug something using
            member names as a shortcut to layer_state.member, so this method checks if the attribute we're
            trying to change is in a layer_state of an initialized Layer.

        Args:
            key:
            value:

        Returns:

        """
        if hasattr(self, "layer_state") and self.layer_state is not None:
            if hasattr(self.layer_state, key):
                setattr(self.layer_state, key, value)

        super(Layer, self).__setattr__(key, value)

    @property
    def trainable_variables(self):
        variables = self.layer_state.variables()
        return [var for var in variables if var.trainable]

    @property
    def variables(self):
        return self.layer_state.variables()

    @classmethod
    def proto(cls, **kwargs):
        return LayerProto(cls, **kwargs)

    def reuse_with(self, *layers, **kwargs):
        kwargs["share_state_with"] = self
        return self.proto(*layers, **kwargs)

    @property
    def input_layers(self):
        return list(self._input_layers)

    @input_layers.setter
    def input_layers(self, input_layers):
        raise ValueError("input_layers can't be set")

    def compute(self, *args):
        raise NotImplementedError("computation not implemented for this layer")

    def __call__(self, *input_layers):
        if not input_layers:
            if self.input_layers:
                ord_inputs = {out: i for i, out in enumerate(self.input_graph.out_nodes)}
                results = self.input_graph()
                input_tensors = tuple([results[ord_inputs[out]] for out in self.input_layers])
            else:
                input_tensors = tuple()
        else:
            input_layers = map(as_layer, input_layers)
            input_tensors = Graph.eval(*input_layers)

            # return {f"{self.scoped_name}_output": self.compute(*input_tensors)}
        return self.compute(*input_tensors)

    def as_function(self, name="layer_function"):
        graph = Graph.build(inputs=None, outputs=self)
        return graph.as_function(fn_name=name, as_tf_function=False)

    def _list_functions_for_serialization(self, serialization_cache):
        concrete_call = tf.function(self.as_function()).get_concrete_function()
        fns = dict({"__call__": concrete_call})
        fns.update(super()._list_functions_for_serialization(serialization_cache))
        return fns

    def _list_extra_dependencies_for_serialization(self, serialization_cache):
        """ Lists extra dependencies to serialize.

        Internal sub-classes can override this method to return extra dependencies

        Args:
             serialization_cache: A dictionary shared between all objects in the same
                object graph. This object is passed to both
                `_list_extra_dependencies_for_serialization` and
                `_list_functions_for_serialization`.

        Returns:
            A dictionary mapping attribute names to trackable objects.
        """
        if self._input_layers:
            layers = self.input_graph.dependency_iter()

            dependencies = {
                f"input_layer_{index}": dep_layer
                for index, dep_layer in enumerate(layers)
            }
        else:
            dependencies = {}
        return dependencies

    def __str__(self):
        """ Informal string representation for a layer consists of Layer Class name, number of units and if its
        Sparse or Dense.

        Returns:
            :obj:`str`: a :obj:`str` with the informal representation for a layer instance.

        """
        class_name = type(self).__name__
        inputs = ",".join(map(lambda x: x.name, self.input_layers))
        return f"{self.scoped_name}::{class_name}({self.n_units},{self.dtype})({inputs})"

    # def __repr__(self):
    #     """ returns
    #
    #     Returns:
    #
    #     """
    #     proto = self.proto
    #     kw = ', '.join([f'{k}={repr(v)}' for k, v in proto.arg_dict.items()])
    #     return f"{proto.layer_cls.__name__}({kw})"

    def __getitem__(self, item):
        if isinstance(item, tf.Tensor):
            item_name = item.op.name
        else:
            item_name = str(item)
        return Lambda(self,
                      fn=lambda output_tensor: output_tensor[item],
                      n_units=self.n_units,
                      dtype=self.dtype,
                      name=f"get_item_{item_name.replace('-', 'minus')}")

    # def __add__(self, other):
    #    other = as_layer(other, dtype=self.dtype)
    #    return Add(self, other)

    def __mul__(self, other):
        other = as_layer(other, dtype=self.dtype)
        return Lambda(self, other, fn=tf.multiply, n_units=self.n_units, name="Mul")


class Lambda(Layer):
    """ Custom Function Layer
    Attributes:
        tensor: the tensor to be wrapped by this layer
        var_list: if vars are involved in the output tensor, they can be specified here
        and will be listed in variables
        n_units: number of units for this layer,
        batch_size: Optional batch size for this layer
        apply_to_layer: if False applies the function to the tensors otherwise applies to the layer
    Creates a layer from a given tensor that one can then integrate with other layers
    """

    def __init__(self, *layers,
                 fn,
                 n_units=None,
                 var_list=None,
                 dtype=None,
                 name="fn_layer",
                 apply_to_layer=False):

        if isinstance(fn, LayerProto):
            raise TypeError("cannot pass a LayerProto to Lambda Layer, pass a callable function instead")
        elif not hasattr(fn, "__call__"):
            raise TypeError("fn must be a callable function")

        super().__init__(input_layers=layers,
                         n_units=n_units,
                         dtype=dtype,
                         name=name,
                         fn=fn,
                         var_list=as_list(var_list),
                         apply_to_layer=apply_to_layer)

    def init_state(self):
        layer_state = super().init_state()
        for i, var in enumerate(self.var_list):
            setattr(layer_state, f"var_{i}", var)
        return layer_state

    def compute(self, *input_tensors):
        with layer_scope(self):
            output = self.fn(*input_tensors)

            # TODO layers are supposed to output only one tensor or sparse tensor
            if self.dtype is not None and output.dtype != self.dtype and not isinstance(output, tf.Operation):
                output = tf.cast(output, self.dtype)

            if self.dtype is None:
                self.dtype = output.dtype if not isinstance(output, tf.Operation) else None

            return output

    def reuse_with(self, *layers, name=None):
        return Lambda(*layers,
                      fn=self.fn,
                      n_units=self.n_units,
                      var_list=self.var_list,
                      dtype=self.dtype,
                      name=self.name,
                      apply_to_layer=self.apply_to_layer)


class ToSparse(Layer):
    """ Transforms the previous layer into a sparse layer.

    This means that the current layer.tensor is a ``SparseTensor``
    """

    def __init__(self, input_layer):
        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=input_layer.dtype,
                         name=input_layer.name + "_sparse")

    def compute(self, input_tensor):
        with layer_scope(self):
            if isinstance(input_tensor, tf.SparseTensor):
                return input_tensor
            else:
                return txf.to_sparse(input_tensor)

    def reuse_with(self, input_layer):
        return ToSparse(input_layer)


class ToDense(Layer):
    """ ToDense transformation layer

    Transforms the previous layer into a dense layer (outputting a dense tensor)
    if the previous layer is already a dense layer, forwards the previous layer doing nothing

    """

    def __init__(self, input_layer):
        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=input_layer.dtype,
                         name=input_layer.name + "_dense")

    def compute(self, input_tensor):
        with layer_scope(self):
            if isinstance(input_tensor, tf.SparseTensor):
                return tf.sparse.to_dense(input_tensor)
            else:
                return input_tensor

    def reuse_with(self, input_layer):
        return ToDense(input_layer)


class Wrap(Layer):
    """ Wraps another layer with tf code

    Utility layer used to wrap arbitrary layers with another op this might be useful to modify
    existing layers without implementing a new ``Layer``.

    Example::

    You can create nested WrapLayers in which case, ``reuse_with`` will replace the inputs
    of the innermost `Layer` (which is not a `WrapLayer`)


                 +------------------------------------+
                 | +-------------------------+        |
                 | | +------------+          |        |
                 | | |            |          |        |
                 | | |   INPUT    | WRAP     | WRAP   |
        +--------------> LAYER    |          |        +------->
                 | | |            |          |        |
                 | | +------------+          |        |
                 | +-------------------------+        |
                 +------------------------------------+


    Attributes:
        wrap_fn


    Args:
        wrapped_layer: a `Layer` to be wrapped by this Layer
        n_units: the new number of units this layer will have
        wrap_fn: a callable returning a `Layer`
        name: name for this layer, defaults to wrap_[layer]
    """

    def __init__(self,
                 wrapped_layer,
                 wrap_fn,
                 n_units=None,
                 forward_attributes=None,
                 name=None):
        forward_attributes = as_list(forward_attributes)
        for attr in forward_attributes:
            if hasattr(wrapped_layer, attr):
                setattr(self, attr, getattr(wrapped_layer, attr))

        super().__init__(input_layers=wrapped_layer.input_layers,
                         n_units=n_units,
                         dtype=wrapped_layer.dtype,
                         name=f"wrap_{wrapped_layer.name}" if name is None else name,
                         wrap_fn=wrap_fn,
                         wrapped=wrapped_layer,
                         forward_attributes=forward_attributes
                         )

    def init_state(self):
        state = super().init_state()
        with layer_scope(self, name=self.name):
            state.wrapped = self.wrapped
            wrap = self.wrap_fn(self.wrapped)
            # to make sure compute applies to the whole function
            state.wrap = Module(inputs=self.wrapped.input_layers,
                                output=wrap)
        return state

    # TODO I think this can be skipped because module is a layer and has variables
    #   needs testing
    @property
    def variables(self):
        return self.layer_state.wrap.variables

    def compute(self, *inputs):
        with layer_scope(self, name=self.name):
            output = self.wrap.compute(*inputs)
            dtype = output.dtype if not isinstance(output, tf.Operation) else None
            fn_n_units = output.get_shape().as_list()[-1]

            if self.n_units is not None and fn_n_units != self.n_units:
                ValueError("provided n_units and result wrap_fn resulting tensor last dimension do not match")
            if self.n_units is None:
                self.n_units = fn_n_units
            if dtype != self.dtype:
                self.dtype = dtype

        return output

    def reuse_with(self, *layers, name=None):
        """ Reuse with a different input layer

            Calls reuse with on the wrapped layer and then creates a new wrapped layer
            around it, using the current tensor function.
        """
        new_wrapped = self.wrapped.reuse_with(*layers)

        # forward any previous attributes if we're wrapping over other WrapLayer instances
        attr_fwd = self.forward_attributes
        if isinstance(new_wrapped, Wrap):
            attr_fwd += new_wrapped.forward_attributes

        if name is None:
            name = self.name

        return Wrap(wrapped_layer=new_wrapped,
                    n_units=self.n_units,
                    wrap_fn=self.wrap_fn,
                    forward_attributes=attr_fwd,
                    name=name)


class Input(Layer):
    """ Input Layer

    Receives input values to a neural network and outputs either ``Tensor`` or ``SparseTensor`` values.

    Input is stateful, which means it will always output the same value until a new value is set:

        ```python
        import tensorx as tx
        # assumes a shape [None,2]
        x = tx.Input(n_units=2, constant=False)
        v1 = x()

        [[0,0]]

        x.value = tf.ones([2,2])
        v2 = x()

        [[1,1],
         [1,1]]

        x.value = tf.ones([2,3])
        # throws an exception because it expects a shape [None,2]

        x2 = tx.Input(n_units=2, constant=True)
        x2()

        [[0,0]]

        x2.value = tf.ones([2,2])
        # throws ValueError: Cannot set the value of a constant Input Layer
        ```

    !!! info
        * when `n_active` is provided, and this layer is connected to an `Linear` layer, this is interpreted as a
        binary sparse (one-hot) input layer and expects it's values to be of type `tf.int64`.

        * `SparseTensor` value can be passed as an initial value

    Args:
        init_value (`Tensor`/`np.ndarray`): initial value for `Input` layer, if given, it determines `n_units`
        n_units (int): number of output units for this layer. Should match last dimension of `init_value` shape and
            `shape` if either is provided.
        n_active (int): number of active units <= n_units. If given, `Input` expects a `Tensor` with indices and
            outputs a sparse array.
        sparse (bool): if true, expects the input value to be a `SparseTensor`.
        shape (List): expected input shape
        dtype (`tf.Dtype`): type for input values.
        constant: if true, input value cannot be changed after `Input` is initialized.
        name (str): layer name
        cast (bool): if True tries to cast the input to the given dtype on value set

    Attributes:
        value (`Tensor`/`SparseTensor`): if `constant=False` this attribute can be set to store a new value in `Input`,
        else, setting `value` will throw an exception.

    """

    def __init__(self,
                 init_value=None,
                 n_units=None,
                 constant=False,
                 sparse=False,
                 n_active=None,
                 shape=None,
                 dtype=None,
                 cast=True,
                 name="input"):

        if n_active is not None and n_active >= n_units:
            raise ValueError("n_active must be < n_units")
        if init_value is not None:
            if n_active is not None:
                self._value = as_tensor(init_value, dtype=tf.int64)
            else:
                self._value = as_tensor(init_value, dtype=dtype)
        else:
            self._value = None
        if self._value is not None:
            dtype = self._value.dtype
        self.init_value = init_value
        self.n_active = n_active
        # otherwise this is assigned as a ListWrapper
        if shape is not None:
            shape = tf.TensorShape(shape)
        self.shape = self._no_dependency(shape)
        self.constant = constant
        self.sparse = sparse
        if dtype is not None:
            self.dtype = dtype
        elif n_active is not None:
            self.dtype = tf.int64
        else:
            self.dtype = tf.float32
        if self.n_active is not None and self.dtype != tf.int64:
            raise TypeError(f"Sparse index input tensors should have type {tf.int64}, {self.dtype} found instead")
        self.n_units = n_units
        # Check params =================================================================================================

        if self.n_units is None and self._value is not None:  # and self.constant:
            # if not constant assume the n_units of the first input it gets
            input_shape = self._value.shape
            self.n_units = input_shape[-1] if len(input_shape) > 0 else 0

        # n_units is None and self._value is not None and not self.constant:
        # in this case we know nothing about the input shape

        if self._value is None:
            if self.shape is None:
                # shape is none and value is None, we have no way to determine the shape of the tensor so we assume 2D
                expected_shape = [None, self.n_units]
            else:
                if self.n_units is not None and self.shape[-1] and self.n_units != self.shape[-1]:
                    raise ValueError(f"n_units and shape[-1] don't match:\n"
                                     f"\t  n_units: {self.n_units}\n\tshape[-1]: {self.shape[-1]}")
                expected_shape = self.shape
        else:  # VALUE NOT NONE
            if self.shape is None:
                # if self.n_units is None at this point it means the value.shape[-1] is None for the input tensor
                if self.n_units is not None and self.n_units > 0:
                    expected_shape = [None] * len(self._value.shape[:-1]) + [self.n_units]
                else:
                    expected_shape = [None] * len(self._value.shape)
            else:
                expected_shape = self.shape

        # n_active overwrites expected shape
        if self.n_active is not None:
            # expecting a sparse 2D Tensor using a dense 2D tensor with indices
            expected_shape = [None, self.n_active]

        expected_shape = tf.TensorShape(expected_shape)

        # verify shape
        if self.shape is not None:
            if not self.shape.is_compatible_with(expected_shape):
                raise ValueError("Invalid shape for Input\n\texpected: {shape}\n\t"
                                 " current: {invalid}".format(shape=expected_shape, invalid=self.shape))
        else:
            self.shape = expected_shape

        # ==============================================================================================================

        super().__init__(None,
                         n_units=self.n_units,
                         dtype=self.dtype,
                         name=name,
                         cast=cast,
                         shape=self.shape)

    def init_state(self):
        layer_state = super().init_state()

        if self._value is None and self.shape[-1] is not None:
            # set the value based on shape
            if self.n_active is None:
                if self.sparse:
                    dense_shape = [1] * len(self.shape[:-1]) + [self.shape[-1]]
                    self._value = txf.empty_sparse_tensor(dense_shape=dense_shape, dtype=self.dtype)
                else:
                    self._value = tf.zeros([1] * len(self.shape[:-1]) + [self.shape[-1]], dtype=self.dtype)
            else:
                self.sparse = True
                self._value = tf.zeros([0] * len(self.shape[:-1]) + [self.shape[-1]], dtype=tf.int64)
        else:
            if isinstance(self._value, tf.SparseTensor):
                self.sparse = True

        if self._value is None:
            # create an empty tensor with a len(self.shape) number of dimensions
            self._value = tf.reshape(tf.constant([],dtype=self.dtype), [0] * len(self.shape))

        with layer_scope(self):
            if not self.constant and self._value is not None:
                if isinstance(self._value, tf.SparseTensor):
                    layer_state.slot = txf.SparseVariable(initial_value=self._value,
                                                          validate_shape=False,
                                                          trainable=False,
                                                          name=f"{self.name}_slot")
                else:
                    layer_state.slot = tf.Variable(initial_value=self._value,
                                                   shape=tf.TensorShape(self.shape),
                                                   dtype=self.dtype,
                                                   validate_shape=False,
                                                   trainable=False,
                                                   name=f"{self.name}_slot")

        return layer_state

    @property
    def value(self):
        """ value

        returns current input value

        Returns:
            value (`Tensor`/`SparseTensor`): input value

        """
        if self.constant:
            return self._value
        else:
            return self.slot.value()

    @value.setter
    def value(self, x):
        if self.constant:
            raise ValueError("Cannot set the value of a constant Input Layer")
        dtype = None if not self.cast else self.dtype
        x = as_tensor(x, dtype)
        if not self.cast and self.dtype is not None and self.dtype != x.dtype:
            raise TypeError(f"Input \"{self.name}\" has dtype {self.dtype}, value received {x.dtype}")

        if not x.shape.is_compatible_with(self.shape):
            raise ValueError(f"Invalid shape:\n"
                             f"\texpected: {self.shape}\n"
                             f"\t current: {x.shape.as_list()}")
        # shape = [d if d is not None else -1 for d in self.shape]
        # x = tf.reshape(x, shape)
        last_dim = x.shape[-1] if len(x.shape) > 0 else 0
        if self.n_active is None and (self.n_units is None or self.n_units != last_dim):
            if len(x.shape) > 1:
                self.n_units = x.shape[-1]
            else:
                raise ValueError("cannot set a layer to a value {} with 0 or 1 dimensions: "

                                 "\n\tsupply a value with dim with at least 2 dimensions (e.g. [[{}]])".format(x, x))

        # validate value
        if self.n_active is not None:
            if not x.shape.is_compatible_with([None, self.n_active]):
                raise ValueError("Invalid shape for Input: expected {shape}".format(shape=[None, self.n_active]))
            # x is interpreted as indices which must be int64
            if x.dtype != tf.int64:
                x = tf.cast(x, tf.int64)
        else:
            if self.shape:
                if len(self.shape) == 0:
                    expected = []
                else:
                    expected = [None] * len(x.shape[:-1].as_list()) + [self.n_units]
                if not x.shape.is_compatible_with(expected):
                    raise ValueError("Invalid shape for Input\n\texpected: {shape}\n\t"
                                     " current: {invalid}".format(shape=expected, invalid=x.shape.as_list()))
            if x.dtype != self.dtype:
                if self.dtype is not None:
                    x = tf.cast(x, self.dtype)
                self.dtype = x.dtype

        # only executed when the previous value was none (we need new cache variables)
        # self.updated = True
        self._value = x
        self.layer_state.slot.assign(x)

    def compute(self):
        with layer_scope(self):
            if self.n_active is not None:
                return tx.sparse_matrix_indices(self.value, num_cols=self.n_units, dtype=self.dtype)
            else:
                return self.value

    def __str__(self):
        class_name = type(self).__name__
        if self.n_active is not None:
            str_representation = f"{self.scoped_name}::{class_name}({self.n_active}/{self.n_units},{self.dtype})[Sparse]"
        else:
            str_representation = f"{self.scoped_name}::{class_name}({self.n_units},{self.dtype})"

        return str_representation

    def reuse_with(self, *layers, name=None):
        raise AttributeError("Cannot call reuse_with on Input Layer: Input has no input layers")


class Param(Input):
    """ Tensor(value) is an alias for Input(value,constant=True)
        """

    def __init__(self,
                 init_value,
                 n_units=0,
                 shape=None,
                 dtype=None,
                 name="param"):
        super().__init__(init_value=init_value,
                         n_units=n_units,
                         constant=False,
                         sparse=None,
                         n_active=None,
                         shape=shape,
                         dtype=dtype,
                         name=name)

        self.observers = []

    @Input.value.setter
    def value(self, value):
        Input.value.fset(self, value)
        for observer in self.observers:
            observer.trigger(OnValueChange(self.name))

    #
    def register(self, observer):
        self.observers.append(observer)


class Constant(Input):
    """ Tensor(value) is an alias for Input(value,constant=True)
    """

    def __init__(self,
                 init_value=None,
                 n_units=None,
                 shape=None,
                 dtype=None,
                 name="tensor"):
        super().__init__(init_value=init_value,
                         n_units=n_units,
                         constant=True,
                         sparse=None,
                         n_active=None,
                         shape=shape,
                         dtype=dtype,
                         name=name)


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
                 shape=None,
                 update_once=False,
                 trainable=False,
                 resource=False,
                 dtype=tf.float32,
                 init=tf.initializers.zeros(),
                 share_state_with=None,
                 name="variable"):

        # these are overwritten by the layer state but are here just for attribute reference
        self.counter = None
        self.variable = None

        self.shape = shape
        self.update_once = update_once
        self.trainable = trainable
        self.resource = resource
        self.init = init
        self.share_state_with = share_state_with

        super().__init__(input_layer,
                         n_units=n_units,
                         dtype=dtype,
                         name=name)

    def init_state(self):
        state = super().init_state()

        input_layer = self.input_layers[-1] if len(self.input_layers) > 0 else None

        if input_layer is not None:
            if self.n_units is not None and self.n_units != input_layer.n_units:
                raise ValueError("n_units must match input_layer.n_units")
            self.n_units = input_layer.n_units
            self.dtype = input_layer.dtype if self.dtype is None else self.dtype
            self.shape = [1, self.n_units]

        else:
            if self.n_units is not None:
                if self.shape is not None:
                    if self.shape[-1] != self.n_units:
                        raise ValueError(
                            f"n_units {self.n_units} does not match var_shape last dimension {self.shape[-1]}")
                else:
                    raise ValueError("shape could not be determined: either supply an input layer or shape")
            else:
                self.n_units = self.shape[-1]

        if self.n_units is None:
            raise ValueError("invalid variable layer parameters: either supply input layer or a valid shape")

        if len(self.shape) > 1:
            self.shape[0] = 1

        with layer_scope(self):
            if self.share_state_with is None:
                variable = tf.Variable(initial_value=self.init(self.shape, dtype=self.dtype),
                                       shape=tf.TensorShape([None] + self.shape[1:]),
                                       validate_shape=True,
                                       trainable=self.trainable,  # default = False
                                       name=self.name + "_variable")

                counter = tf.Variable(initial_value=0,
                                      dtype=tf.int32,
                                      trainable=False,
                                      name="counter")

                state.variable = variable
                state.counter = counter
            else:
                state = self.share_state_with.layer_state

        return state

    def compute(self, *input_tensors):
        input_tensor = input_tensors[0] if input_tensors else None

        with layer_scope(self):
            def update():
                self.layer_state.counter.assign_add(1)
                self.layer_state.variable.assign(input_tensor)
                return self.layer_state.variable.value()

            if input_tensor is not None:
                if self.update_once:
                    return tf.cond(tf.math.less(self.layer_state.counter, 1),
                                   update,
                                   lambda: self.layer_state.variable.value())

                else:
                    return update()
            else:
                return self.layer_state.variable.value()

    def reset(self):
        """ reset

        resets the variable using its initializer

        Returns:
            an op that can be run to reinitialize the variable
        """
        with layer_scope(self):
            self.layer_state.variable.assign(self.init(self.shape))
            self.layer_state.counter.assign(0)

    def reuse_with(self, input_layer=None, init_from_input=None, name=None):
        input_layer = self.input_layers[0] if input_layer is None else input_layer
        name = self.name if name is None else name
        init_from_input = self.update_once if init_from_input is None else init_from_input

        return VariableLayer(input_layer=input_layer,
                             shape=self.shape,
                             update_once=init_from_input,
                             trainable=self.trainable,
                             resource=self.resource,
                             dtype=self.dtype,
                             init=self.init,
                             share_state_with=self,
                             name=name
                             )


class Transpose(Layer):
    """ Transpose

        Args:
            input_layer (Layer):
    """

    def __init__(self, input_layer, perm=None, n_units=None, name="transpose"):

        n_units = n_units if perm is None or perm[-1] != len(perm) - 1 else input_layer.n_units

        super().__init__(input_layers=input_layer,
                         n_units=n_units,
                         dtype=input_layer.dtype,
                         name=name,
                         perm=perm)

    def compute(self, *input_tensors):
        input_tensor = input_tensors[0]

        with layer_scope(self):
            if not isinstance(input_tensor, tf.SparseTensor):
                output = tf.transpose(input_tensor, self.perm)
            else:
                output = tf.sparse.transpose(input_tensor, self.perm)

            # this is only needed to determine the
            n_units = output.shape.as_list()[-1]

            if self.n_units is not None:
                if self.n_units != n_units:
                    expected = self.n_units
                    raise ValueError("n_units is different from defined n_units:\n"
                                     f"\texpected: {expected}\n"
                                     f"\tn_units: {n_units}")

            return output

    def reuse_with(self, input_layer, name=None):
        if name is None:
            name = self.name
        return Transpose(input_layer=input_layer, perm=self.perm, name=name)


class Reshape(Layer):
    """

    Args:
        input_layer (Layer): an input layer to be reshaped
    """

    def __init__(self, input_layer, shape, name="reshape"):

        self.shape = shape
        self.target_shape = [d if d is not None else -1 for d in shape]
        n_units = self.target_shape[-1] if self.target_shape[-1] > 0 else None

        super().__init__(input_layers=input_layer,
                         n_units=n_units,
                         dtype=input_layer.dtype,
                         name=name)

    def compute(self, *input_tensors):
        input_tensor = input_tensors[0]

        with layer_scope(self):
            if not isinstance(input_tensor, tf.SparseTensor):
                output = tf.reshape(input_tensor, self.target_shape)
            else:
                output = tf.sparse_reshape(input_tensor, self.target_shape)

        # TODO should I really update n_units or make it clear that this should
        #  be undefined if the user doesn't provide it?
        # update n_units
        n_units = output.get_shape().as_list()[-1]
        if self.n_units is None:
            self.n_units = n_units
        elif self.n_units != n_units:
            raise ValueError(
                "n_units changes between computations:\n\tprevious: {}\n\t current: {}".format(self.n_units, n_units))
        return output

    def reuse_with(self, input_layer, name=None):
        if name is None:
            name = self.name
        return Reshape(input_layer, self.target_shape, name)


class Linear(Layer):
    """ Linear(input_layer: Layer, n_units, shape=None add_bias=True)

    Fully connected layer that implements a linear transformation of the form $f(x) = Wx + b$

    Args:
        input_layer (`Layer`): input layer or a value convertible to Layer
        n_units (Optional[int]): number of output units
        shape (Optional[Iterable[int]]): shape for the layer weights (not the output). Needed if n_units and
            input_layer.n_units is not known. If add_bias then the bias variable has shape [shape[-1]]
        weight_init: weights (W) initializer function
        weights: variable to be used as linear weights
        bias: variable to be used as a bias
        transpose_weights: if True, transposes the weights of this layer (must match n_units)
        sparse_weights: if True indicates we are using a sparse tensor instead of a tf.Variable for weights
        add_bias: if True, this layers becomes an affine transformation layer xW+b
        bias_init: bias initializer function
        weight_norm (bool): if True weights are normalised
        dtype (``tf.DType``): type for layer variables
        name (str): layer name
        share_state_with: Linear layer with which we wish to share the state

    """

    def __init__(self,
                 input_layer: Layer,
                 n_units: Optional[int] = None,
                 shape: List[int] = None,
                 weight_init=tf.initializers.glorot_uniform(),
                 weights=None,
                 add_bias=True,
                 bias_init=tf.initializers.zeros(),
                 bias=None,
                 transpose_weights=False,
                 sparse_weights=False,
                 weight_norm=False,
                 dtype=tf.float32,
                 name="linear",
                 share_state_with=None):

        shape = tuple(as_list(shape)) if shape else None

        if not isinstance(input_layer, Layer):
            input_layer = Input(input_layer, constant=True, dtype=dtype)

        if input_layer.n_units is None or isinstance(input_layer.n_units, tf.Tensor):
            if shape is None:
                raise ValueError("Cannot create Linear layer from unknown previous n_units")

        if shape is not None:
            if n_units is None:
                n_units = shape[-1]
            if shape[-1] != n_units:
                raise ValueError("shape[-1] does not match n_units:\n\tshape[-1]: {}"
                                 "\n\tn_units: {}".format(shape[-1], n_units))
            if input_layer.n_units is not None:
                if shape[0] != input_layer.n_units:
                    raise ValueError("shape[0] does not match input.n_units:\n\tshape[0]: {}"
                                     "\n\tinput.n_units: {}".format(shape[0], input_layer.n_units))
        else:
            shape = [input_layer.n_units, n_units]

        self.shape = shape
        self.weight_init = weight_init
        self.weights = weights
        self.add_bias = add_bias
        self.bias_init = bias_init
        self.bias = bias
        self.transpose_weights = transpose_weights
        self.sparse_weights = sparse_weights
        self.weight_norm = weight_norm
        self.share_state_with = share_state_with

        super().__init__(input_layers=input_layer,
                         n_units=n_units,
                         dtype=dtype,
                         name=name
                         )

    def init_state(self):
        with layer_scope(self):
            input_layer = self.input_layers[0]

            # weights_shape = [input_layer.n_units, self.n_units]

            if self.share_state_with is not None:
                if not isinstance(self.share_state_with, Linear):
                    raise TypeError("Layer can only share variables with other layer of the same type")

                layer_state = self.share_state_with.layer_state

                shape = [input_layer.n_units, self.n_units]
                shared_shape = layer_state.weights.get_shape().as_list()
                if self.transpose_weights:
                    shared_shape = shared_shape[::-1]

                if shape != shared_shape:
                    raise ValueError("Can only share variables with layers with the same dimensions: "
                                     "share_state_with is provided but \n"
                                     "self shape: {s0} different from "
                                     "other shape: {s1}".format(s0=shape, s1=shared_shape))
            else:
                layer_state = super().init_state()

            # if weights are passed, check that their shape matches the layer shape
            if self.weights is not None:
                weight_shape = self.weights.shape

                if self.transpose_weights:
                    if not tf.TensorShape([input_layer.n_units]).is_compatible_with(tf.TensorShape([weight_shape[-1]])):
                        raise ValueError(
                            "weight shape mismatch: \n\tinput_layer.n_units: {}\n\tself.n_units:{}\n\t"
                            "with transpose_weights=True".format(
                                input_layer.n_units,
                                weight_shape[-1]))
                else:
                    if not tf.TensorShape([input_layer.n_units]).is_compatible_with(tf.TensorShape([weight_shape[0]])):
                        raise ValueError(
                            "weight shape mismatch: input_layer shape {} :: weights shape {} "
                            "with transpose_weights=False".format(
                                input_layer.shape,
                                weight_shape))

            if self.bias is not None:
                bias_shape = self.bias.get_shape().as_list()
                if bias_shape[0] != self.n_units:
                    raise ValueError(
                        f"invalid shared bias: number of bias {bias_shape[0]} "
                        f"does not match number of units {self.n_units}")

            # weights in layer_state overwrite the weights specified
            # should it not be this way?
            # TODO alternative is to create a layer state that differs from the share_state_with
            #   but this changes the semantics
            weights = getattr(layer_state, "weights", self.weights)
            if weights is None:
                init_value = self.weight_init(self.shape, dtype=self.dtype)
                weights = tf.Variable(initial_value=init_value,
                                      trainable=True,
                                      dtype=self.dtype,
                                      name="weights")

            if not hasattr(layer_state, "weights"):
                layer_state.weights = weights

            bias = getattr(layer_state, "bias", self.bias)
            if self.add_bias:
                n_units = self.n_units if self.n_units is not None else tf.shape(weights)[-1]
                if bias is None:
                    bias = tf.Variable(initial_value=self.bias_init([n_units], self.dtype),
                                       name="bias", trainable=True)

            if not hasattr(layer_state, "bias"):
                layer_state.bias = bias

        return layer_state

    def compute(self, input_tensor):
        input_tensor = as_tensor(input_tensor)
        weights = self.layer_state.weights
        if input_tensor.dtype != weights.dtype:
            raise ValueError(f"invalid dtype for Linear inputs:\n"
                             f"\t expected (weights dtype): {weights.dtype}\n"
                             f"\t                 received: {input_tensor.dtype}")

        with layer_scope(self):

            if self.weight_norm:
                weights = tf.math.l2_normalize(weights, axis=[0])

            # y = xW
            if isinstance(input_tensor, tf.SparseTensor):
                sp_values = input_tensor

                # if we use shared weights that must be transposed
                # but we have a sparse input to this layer, this is the most efficient way to do it
                if self.transpose_weights:
                    dense_sp = tf.sparse.to_dense(sp_values)
                    lookup_sum = tf.math.sparse_matmul(a=dense_sp,
                                                       b=weights,
                                                       a_is_sparse=True,
                                                       b_is_sparse=self.sparse_weights,
                                                       transpose_b=True)
                else:

                    sp_indices = txf.sparse_indices(sp_values)
                    # TODO I complained before about this being optimized for distributed TF because
                    #  gradients cannot propagate through gather
                    #  CHECK IF this is still the case in TF2
                    # tf.nn.embedding_lookup_sparse
                    lookup_sum = embedding_lookup_sparse(params=weights,
                                                         sp_tensor=sp_values,
                                                         # sp_ids=sp_indices,
                                                         # sp_weights=sp_values,
                                                         combiner="sum",
                                                         name=self.scoped_name + "_embeddings")
                tensor = lookup_sum
            else:
                input_shape = input_tensor.get_shape().as_list()
                rank = len(input_shape)
                if rank > 2:
                    if self.transpose_weights:
                        axes = [[rank - 1], [1]]
                    else:
                        axes = [[rank - 1], [0]]
                    # Broadcasting is required for the inputs.
                    tensor = tf.tensordot(a=input_tensor,
                                          b=weights,
                                          axes=axes)
                    # Reshape the output back to the original ndim of the input.
                    if not tf.executing_eagerly():
                        output_shape = input_shape[:-1] + [self.n_units]
                        tensor.set_shape(output_shape)
                else:
                    tensor = tf.matmul(a=input_tensor,
                                       b=weights,
                                       name="mat_mul",
                                       transpose_b=self.transpose_weights,
                                       b_is_sparse=self.sparse_weights)

            # y = xW + [b]
            if self.add_bias:
                tensor = tf.nn.bias_add(tensor, self.bias, name="add_b")

        return tensor

    def reuse_with(self, input_layer, name=None, transpose_weights=None, sparse_weights=None):
        """ Reuses the current layer on a different input.

        Uses the variables in this layer to create a new Layer instance with a different input_layer

        Args:
            sparse_weights: if True it means the weights of this linear layer are sparse (usually provided as shared weights)
            transpose_weights: if True transposes the self.weights. Useful to share the state with other layer
            input_layer: a ``Linear` layer
            name: name for the new ``Layer``

        Returns:
            layer (Layer): a new layer with shared variables with the current layer.

        """
        # if current layer is sharing variables, forward the sharing
        share_state_with = self if self.share_state_with is None else self.share_state_with

        if name is None:
            name = self.name

        if transpose_weights is None:
            transpose_weights = self.transpose_weights
        if sparse_weights is None:
            sparse_weights = self.sparse_weights

        return Linear(input_layer=input_layer,
                      n_units=self.n_units,
                      weight_init=self.weight_init,
                      weights=self.weights,
                      transpose_weights=transpose_weights,
                      sparse_weights=sparse_weights,
                      add_bias=self.add_bias,
                      weight_norm=self.weight_norm,
                      name=name,
                      share_state_with=share_state_with)


class Module(Layer):
    """ Module Layer

    Creates a single ``Layer`` object from an existing graph defined between a set of inputs and a single output layer.
    The resulting object can be reused with other inputs just like any other `Layer`.

    !!! note "Note"
        The difference between a `Module` and a `Graph` is that a `Module` has a single output, and can be used as a
        `Layer`. A layer graph is a utility to group multiple layers into a single function &mdash;possibly with
         multiple outputs.

    Raises:
        ValueError: is raised if an input layer does not connect to the output. A ``Module`` needs to be a
        self-contained layer graph.

    Args:
        inputs : one or more input layers
        output : output layer; if no inputs are passed, it follows the graph backwards from the output
        missing_inputs: if True adds missing inputs to the graph but only tries to feed it the given inputs
    """

    def __init__(self, inputs, output=None, dependencies=[], name="module"):
        self.inputs = as_list(inputs)
        self.output = output
        self.dependencies = as_list(dependencies)

        try:
            inputs = self.inputs + self.dependencies
            self.graph = Graph.build(inputs=inputs, outputs=self.output, add_missing_inputs=True)

            # required for Modules built with inputs=None
            # this will make sure that module inputs are always computed
            # before module_fn is computed
            inputs = self.graph.in_nodes
            # to that we don't need to call reuse on compute with params
            self.module_fn = self.graph.as_function(ord_inputs=inputs,
                                                    ord_outputs=output,
                                                    fn_name=f"{name}_fn")
        except ValueError as e:
            raise ValueError(f"Could not build a module with the given "
                             f"endpoints: \n\t{str(e)}")

        super().__init__(input_layers=self.graph.in_nodes,
                         n_units=output.n_units,
                         dtype=output.dtype,
                         name=name,
                         inputs=self.inputs,
                         dependencies=self.dependencies)

    def init_state(self):
        state = super().init_state()

        # add layer to state so that we can retrieve variables of a module etc
        for i, node in enumerate(self.graph.nodes):
            setattr(state, f"layer_{i}", node)
        return state

    def compute(self, *input_tensors):
        input_tensors = map(as_tensor, input_tensors)
        return self.module_fn(*input_tensors)

    def reuse_with(self, *input_layers, name=None):
        if name is None:
            name = self.name

        nl = len(input_layers)
        # nm = len(self.input_layers)
        nm = len(self.inputs)

        if nl > nm:
            raise ValueError(f"Module has {nm} input layers, {nl} provided")

        input_layers = [as_layer(lr) for lr in input_layers]
        # we only match reuse_with input_layers with module inputs, although other dependencies might exist
        matching_inputs = self.inputs[:nl]
        other_inputs = self.inputs[nl:]
        mismatch_dtype = list(map(lambda x: x[0].dtype != x[1].dtype, zip(input_layers, matching_inputs)))
        input_layers = input_layers + other_inputs
        if any(mismatch_dtype):
            raise ValueError(f"dtype mismatch in reuse_with:\n"
                             f"\t     expected types: {[lr.dtype for lr in self.input_layers]}\n"
                             f"\t calling reuse with: {[lr.dtype for lr in input_layers]}")

        # map from self.input_layers[0] -> new_input_layers[0] ...
        layer_map = dict(zip(self.input_layers, input_layers + self.dependencies))

        dep_iter = self.graph.dependency_iter()
        for node in dep_iter:
            if node not in self.graph.in_nodes:
                new_inputs = [layer_map[lr] for lr in node.input_layers]  # if lr not in other_inputs]
                layer_map[node] = node.reuse_with(*new_inputs)

        new_output = layer_map[self.output]

        return Module(inputs=input_layers, output=new_output, dependencies=self.dependencies, name=name)


class ViewLayer(Layer):
    """ ViewLayer

    Has same shape and inputs as input layer and stores this layer for future reference.
    This means ViewLayer can substitute Layer where layer would be used

    Properties:
        inner_layer (Layer) wrapped by a view
    """

    def __init__(self, layer, dtype=None, forward_attributes=None, name=None, **kwargs):
        self.inner_layer = layer

        super().__init__(input_layers=layer.input_layers,
                         n_units=layer.n_units,
                         dtype=layer.dtype if dtype is None else dtype,
                         name=f"view_{layer.name}" if name is None else name,
                         **kwargs)

        self.attr_fwd = as_list(forward_attributes)
        for attr in self.attr_fwd:
            if hasattr(self.inner_layer, attr):
                setattr(self, attr, getattr(self.inner_layer, attr))

    def compute(self, *input_tensors):
        return super().compute(*input_tensors)

    def init_state(self):
        layer_state = super().init_state()
        setattr(layer_state, "inner_layer", self.inner_layer)
        return layer_state


class DropConnect(ViewLayer):
    """ DropConnect

    Wraps around a Linear layer to create a new layer where connections between input and linear units
    can be dropped with a given probability.

    Note:
        as opposed to ``Dropout`` which does not wrap a layer but rather drops the outputs of it's previous
        layer.

    Args:
            input_layer (Linear): ``Linear`` layer to be wrapped in DropConnect
            probability (float): probability of dropping a connection between units
            locked (bool): if true
            name (str):

    """

    def __init__(self, input_layer, probability=0.5, locked=True, share_state_with=None, name=None):
        if not isinstance(input_layer, Linear):
            raise TypeError(f"DropConnect can only wrap Linear layers: {input_layer} found instead")

        self.bias = None
        self.weights = None
        self.probability = probability
        self.locked = locked
        self.share_state_with = share_state_with

        super().__init__(input_layer,
                         name=name if name is not None else f"drop_{input_layer.name}"
                         )

    def init_state(self):
        if self.share_state_with is not None:
            layer_state = self.share_state_with.layer_state
        else:
            layer_state = super().init_state()
            layer_state.weight_mask = txf.binary_mask(self.inner_layer.weights, self.probability)
            if self.inner_layer.bias is not None:
                layer_state.bias_mask = txf.binary_mask(self.inner_layer.bias)

        return layer_state

    def compute(self, input_tensor):
        with layer_scope(self):
            weights = self.inner_layer.weights
            bias = self.inner_layer.bias

            drop_w, w_mask = txf.dropout(weights, probability=self.probability,
                                         random_mask=self.layer_state.weight_mask,
                                         scale=False,
                                         return_mask=True)
            # self.layer_state.weight_mask = w_mask
            drop_bias = None
            add_bias = bias is not None
            if add_bias:
                drop_bias, b_mask = txf.dropout(bias,
                                                probability=self.probability,
                                                random_mask=self.layer_state.bias_mask,
                                                scale=False,
                                                return_mask=True)
                # self.layer_state.bias_mask = b_mask

            new_linear = Linear(input_tensor,
                                n_units=self.n_units,
                                weights=drop_w,
                                bias=drop_bias,
                                add_bias=add_bias)
            # forward weights and bias
            # TODO problem with this is that it creates a conditional compute function that alters the state of its
            #  container
            self.weights = new_linear.weights
            self.bias = new_linear.bias

            return new_linear.compute(input_tensor)

    def reuse_with(self, input_layer, name=None, locked=None):
        new_layer = self.inner_layer.reuse_with(input_layer)

        locked = self.locked if locked is None else locked
        name = self.name if name is None else name
        share_state_with = self if locked else None

        return DropConnect(input_layer=new_layer,
                           probability=self.probability,
                           locked=locked,
                           share_state_with=share_state_with,
                           name=name)


class LayerNorm(Layer):
    """ Layer Normalization

    !!! cite "Reference"
        Lei Ba J., Kiros, J., Hinton, G. [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016

    Args:
        input_layer (`Layer`): the layer to be normalized
    """

    def __init__(self, input_layer, share_state_with=None):
        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=tf.float32,
                         name="layer_normalization",
                         share_state_with=share_state_with
                         )

    def init_state(self):
        if self.share_state_with is None:
            state = super().init_state()
            # center
            state.bias = tf.Variable(tx.zeros_init()([self.n_units]), trainable=True, name="beta")
            # scale
            state.scale = tf.Variable(tx.ones_init()([self.n_units]), trainable=True, name="gamma")
        else:
            state = self.share_state_with.layer_state

        return state

    def compute(self, *input_tensors):
        input_tensor = input_tensors[0]

        with layer_scope(self):
            mean, variance = tf.nn.moments(input_tensor, -1, keepdims=True)
            variance_epsilon = 1e-12

            return tf.nn.batch_normalization(
                input_tensor,
                mean,
                variance,
                offset=self.bias,
                scale=self.scale,
                variance_epsilon=variance_epsilon)


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
        probability: a scalar float with the probability that each element is dropped.
        alpha (`Bool`): if True uses alpha dropout instead of dropout, default is False.
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
                 mask=None,
                 noise_shape=None,
                 locked=False,
                 alpha=False,
                 seed=None,
                 name="dropout",
                 share_state_with=None
                 ):

        self.probability = probability
        self.scale = scale
        self.mask = mask
        self.noise_shape = noise_shape
        self.locked = locked
        self.seed = seed
        self.share_state_with = share_state_with
        self.alpha = alpha

        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=input_layer.dtype,
                         name=name
                         )

    def init_state(self):
        input_layer = self.input_layers[0]
        mask = as_layer(self.mask) if self.mask is not None else None

        # random mask layer
        @layer(n_units=input_layer.n_units, name="random_mask")
        def random_mask(x):
            noise_shape = tf.shape(x)
            dtype = x.dtype

            keep_prob = 1 - self.probability
            random_state = tf.random.uniform(noise_shape,
                                             seed=self.seed,
                                             dtype=dtype)
            mask_tensor = keep_prob + random_state
            mask_tensor = tf.math.floor(mask_tensor, name="binary_mask")
            return mask_tensor

        if self.share_state_with is None:
            layer_state = super().init_state()
            if mask is not None:
                layer_state.mask = mask
            else:
                with layer_scope(self):
                    if self.locked:
                        layer_state.mask = random_mask(input_layer)
                    # else:
                    #    self.layer_state.mask = None
        else:
            layer_state = self.share_state_with.layer_state

        # note usually we don't manipulate input_layers directly, but the random mask is a new dependency
        if hasattr(layer_state, "mask"):
            self._input_layers.append(layer_state.mask)

        return layer_state

    def compute(self, *input_tensors):
        """
        x, mask = input_tensors

        Args:
            *input_tensors:

        Returns:

        """
        input_value = input_tensors[0]
        if len(input_tensors) > 1:
            mask = input_tensors[-1]
        else:
            if hasattr(self.layer_state, "mask"):
                mask = self.layer_state.mask
            else:
                mask = None

        # mask_value = mask.compute() if mask is not None else None
        mask_value = mask

        with layer_scope(self):
            if isinstance(input_value, tf.SparseTensor):
                # if input is sparse, noise_shape is not used
                tensor, mask = txf.sparse_dropout(sp_tensor=input_value,
                                                  mask=mask_value,
                                                  probability=self.probability,
                                                  scale=self.scale,
                                                  return_mask=True,
                                                  alpha=self.alpha,
                                                  seed=self.seed)

            else:
                if self.alpha:
                    tensor, mask = txf.alpha_dropout(tensor=input_value,
                                                     noise_shape=self.noise_shape,
                                                     random_mask=mask_value,
                                                     probability=self.probability,
                                                     return_mask=True,
                                                     seed=self.seed)
                else:
                    tensor, mask = txf.dropout(tensor=input_value,
                                               noise_shape=self.noise_shape,
                                               random_mask=mask_value,
                                               probability=self.probability,
                                               scale=self.scale,
                                               return_mask=True,
                                               seed=self.seed)

            return tensor

    def reuse_with(self, input_layer, mask=None, name=None, locked=None):
        locked = self.locked if locked is None else locked
        name = self.name if name is None else name
        share_state_with = self if locked else None
        mask = self.mask if mask is None else mask

        return Dropout(input_layer,
                       mask=mask,
                       probability=self.probability,
                       noise_shape=self.noise_shape,
                       scale=self.scale,
                       locked=locked,
                       seed=self.seed,
                       alpha=self.alpha,
                       share_state_with=share_state_with,
                       name=name)


class DropLookup(Layer):
    """ Applies Dropout with a given probability to a Lookup layer by unique id

    A lookup represents a sequence in batch-major form
    For each lookup sample, the dropout is applied per unique id, meaning that if
    the lookup ids in a sample are [1,2,1] and id 1 is selected for dropout,
    the first and last vectors will be set to 0 for this sample.

    !!! warning
        contrary to Dropout, this lookup cannot be shared, meaning that reuse_with will just apply a new random mask to
        the new Lookup layer, instead of sharing the same mask.

    Args:
        lookup_layer: a Lookup input layer
        scale: if scale is True, scales the non-dropped lookups by x / (1 - probability)


    """

    def __init__(self, lookup_layer, indices=None, probability=0.5, scale=True, name="drop_lookup"):
        if not isinstance(lookup_layer, Lookup):
            raise TypeError("input layer should be a {} layer {} found instead".format(str(Lookup), type(lookup_layer)))

        # DropLookup gets the input layer of the lookup layer to get it's indices
        # because it's a lookup layer, it must have indices
        if indices is None:
            indices = lookup_layer.input_layers[0]

        self.scale = scale
        self.probability = probability
        super().__init__(input_layers=[lookup_layer, indices], n_units=lookup_layer.n_units, name=name)

    def compute(self, input_value, indices):
        if self.probability == 1:
            return tf.zeros_like(input_value, dtype=tf.float32)
        elif self.probability > 0:
            lookup_shape = tf.shape(input_value)
            batch_size, seq_size = lookup_shape[0], lookup_shape[1]

            if isinstance(indices, tf.SparseTensor):
                _, ids = tf.split(indices.indices, 2, axis=-1)
            else:
                ids = indices

            unique_ids, indices = tf.unique(tf.reshape(ids, [-1]))
            mask_shape = tf.stack([batch_size, tf.size(unique_ids)])
            unique_mask = tf.random.uniform(mask_shape, dtype=tf.float32)

            batch_wise = tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=-1),
                                         tf.stack([batch_size, seq_size]))
            unique_batch_wise = tf.reshape(indices, [batch_size, seq_size])

            # gather mask and convert it to binary mask
            mask_indices = tf.stack([batch_wise, unique_batch_wise], axis=-1)
            binary_mask = tf.floor(tf.gather_nd(unique_mask, mask_indices) + (1 - self.probability))
            if self.scale:
                binary_mask /= (1 - self.probability)

            dropped_lookup = input_value * tf.expand_dims(binary_mask, axis=-1)
        else:
            dropped_lookup = input_value

        return dropped_lookup


class BaseRNNCell(Layer, ABC):
    """

    Args:
        input_layer the input Layear for this cell
        previous_state: the recurrent input Layer for the cell
        state_size: list of number of units for each element in the state, default is a single state with [n_units]
        n_units: number of activation units for the RNN cell
        dtype: Layer (output) dtype
    """

    @staticmethod
    def zero_state(n_units, stateful=True, name="zero_state"):
        if stateful:
            # init only once from zero state
            zero_state = VariableLayer(  # input_layer=zero_state,
                shape=[1, n_units],
                n_units=n_units,
                name=name)

        return zero_state

    def __init__(self,
                 input_layer,
                 previous_state,
                 state_size,
                 n_units,
                 dtype=tf.float32,
                 w_init=tf.initializers.glorot_uniform(),
                 u_init=tf.initializers.glorot_uniform(),
                 bias_init=tf.initializers.zeros(),
                 activation=tf.tanh,
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

        self.state_size = state_size

        def init_states(enum_state):
            i, state = enum_state
            if state is None:
                return BaseRNNCell.zero_state(state_size[i])
            else:
                if isinstance(state, Layer):
                    if state.n_units != state_size[i]:
                        raise ValueError(
                            "previous state {i} n_units ({n_units}) not compatible with "
                            "state n_units ({state_shape})".format(
                                i=i,
                                n_units=state.n_units,
                                state_shape=state_size[i]))
                else:
                    state = Input(state, n_units=state_size[i], constant=True)
                return state

        if previous_state is not None:
            previous_state = as_list(previous_state)
            if len(previous_state) != len(state_size):
                raise ValueError(
                    "previous state should have {} states: {} passed instead".format(len(state_size),
                                                                                     len(previous_state)))
        else:
            previous_state = tuple([None] * len(state_size))

        # feel previous states
        # TODO VALIDATION I should really validate the previous_state object that we receive
        # if we receive a list of lists of layers, it fails
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
        self.bias_init = bias_init
        self.activation = activation

        # util class to apply regularizers or forward views
        class Regularizer:
            def __init__(self, func: LayerProto):
                self.func = func
                self.reg = None

            def reset(self):
                self.reg = None

            def __call__(self, *input_layers):

                reg_layers = []
                if self.reg is None:
                    self.reg = []
                    for in_layer in input_layers:
                        reg_layer = self.func(in_layer)
                        reg_layers.append(reg_layer)

                        if issubclass(self.func.layer_cls, ViewLayer):
                            self.reg.append(lambda x: x)
                        else:
                            # dropouts we can re-use
                            self.reg.append(reg_layer.reuse_with)
                else:
                    for in_layer, reg in zip(input_layers, self.reg):
                        reg_layers.append(reg(in_layer))

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

        # needs to be defined on each recurrent cell just as we define self.compute()
        # the default state is the current cell which gives access to its  output tensor
        # self.layer_state =self

        super().__init__(input_layers=[input_layer] + self.previous_state,
                         n_units=n_units,
                         dtype=dtype,
                         name=name)

    def reuse_with(self, input_layer, *previous_state, regularized=None, name=None, **kwargs):
        # because we use objects and not scopes we can use self always on share state with
        share_state_with = self  # self if self.share_state_with is None else self.share_state_with
        previous_state = self.previous_state if len(previous_state) == 0 else previous_state
        name = self.name if name is None else name
        regularized = self.regularized if regularized is None else regularized

        return type(self)(
            input_layer=input_layer,
            n_units=self.n_units,
            previous_state=previous_state,
            activation=self.activation,
            bias_init=self.bias_init,
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
                 previous_state=None,
                 cell_proto: Callable[[Union[Layer, tf.Tensor]], BaseRNNCell] = None,
                 n_units=None,
                 reverse=False,
                 regularized=False,
                 stateful=False,
                 return_state=False,
                 name="rnn_layer",
                 share_state_with: Optional['RNN'] = None
                 ):

        if not cell_proto and not n_units:
            raise ValueError("cell proto and n_units cannot both be None")
        else:
            if not cell_proto:
                cell_proto = RNNCell.proto(n_units=n_units)

        # self.cell_proto = cell_proto
        self.cell = None
        # self.regularized = regularized
        # self.reverse = reverse
        self.previous_state = previous_state
        # self.stateful = stateful
        # self.return_state = return_state
        # self.share_state_with = share_state_with

        # n_units and shape are set after the first cell is created
        super().__init__(input_layers=[input_seq] + as_list(previous_state),
                         n_units=cell_proto.arg_dict["n_units"],
                         dtype=tf.float32,
                         name=name,
                         cell_proto=cell_proto,
                         regularized=regularized,
                         reverse=reverse,
                         stateful=stateful,
                         return_state=return_state,
                         share_state_with=share_state_with)

    def init_state(self):
        layer_state = super().init_state()
        input_seq = self.input_layers[0]

        with layer_scope(self):
            # TODO add input_dim to RNNCells for syntax sugar
            #  create dummy input which is used to init the cell init state without running the entire graph
            #  I guess computing output shape would be useful here
            x0 = tf.ones_like(input_seq[0])

            if self.share_state_with is not None:
                cell = self.share_state_with.cell
                cell = cell.reuse_with(input_layer=x0,
                                       # previous_state=self.previous_state,
                                       regularized=self.regularized)
            else:
                cell = self.cell_proto(x0, previous_state=self.previous_state)
                if cell.regularized != self.regularized:
                    # create a new regularized cell if somehow the regularized parameter doesn't match the constructor
                    cell = cell.reuse_with(input_layer=x0,
                                           # previous_state=self.previous_state,
                                           regularized=self.regularized)

            layer_state.cell = cell
            if self.previous_state is None:
                self.previous_state = cell.previous_state
                # if no previous state is provided we need to add it from current cell
                self._input_layers += as_list(self.previous_state)
            self.n_units = cell.n_units

        return layer_state

    def compute(self, input_seq, *prev_state):
        with layer_scope(self):
            seq_len = tf.shape(input_seq)[0]
            input_ta = tf.TensorArray(dtype=input_seq.dtype, size=seq_len, tensor_array_name="inputs",
                                      clear_after_read=False)
            input_ta = input_ta.unstack(input_seq)
            output_ta = tf.TensorArray(dtype=self.dtype, size=seq_len, tensor_array_name="outputs")
            # state_ta = tf.TensorArray(dtype=self.dtype, size=seq_len, tensor_array_name="states")

            if self.reverse:
                i0 = seq_len - 1
                ii = i0 - 1
                fi = 0
            else:
                i0 = 0
                ii = i0 + 1
                fi = seq_len

            x0 = input_ta.read(i0)
            output_ta = output_ta.write(i0, self.cell.compute(x0, *prev_state))
            state = tuple([state_i.compute(x0, *prev_state) for state_i in self.cell.state])

            # state_ta = state_ta.write(i0, state)

            def rnn_unroll(i, outputs, previous_state):
                xt = input_ta.read(i)
                c = self.cell.compute(xt, *previous_state)
                curr_state = tuple([state_i.compute(xt, *previous_state) for state_i in self.cell.state])

                outputs = outputs.write(i, c)
                if self.reverse:
                    i = i - 1
                else:
                    i = i + 1
                return i, outputs, curr_state

            i, out, last_state = tf.while_loop(cond=lambda i, *_: tf.math.not_equal(i, fi),
                                               body=rnn_unroll,
                                               loop_vars=(ii, output_ta, state),
                                               name="rnn_unroll",
                                               parallel_iterations=1)

            # getting the results and store them in the previous state
            if self.stateful:
                for zero_state, last_state in zip(self.cell.previous_state, last_state):
                    zero_state.variable.assign(last_state)
                out = out.stack()
            else:
                out = out.stack()

            # TODO another solution would be to have a separate var for the last state and another for previous state
            if self.return_state:
                return out, last_state
            else:
                return out

    def reuse_with(self, input_seq, *previous_state, regularized=None, reverse=None, stateful=None,
                   return_state=None, name=None):
        name = self.name if name is None else None
        regularized = self.regularized if regularized is None else regularized
        reverse = self.reverse if reverse is None else reverse
        share_state_with = self.share_state_with if self.share_state_with is not None else self
        previous_state = self.previous_state if not previous_state else previous_state
        return_state = self.return_state if return_state is None else return_state
        stateful = self.stateful if stateful is None else stateful

        return RNN(input_seq=input_seq,
                   previous_state=previous_state,
                   cell_proto=self.cell_proto,
                   regularized=regularized,
                   stateful=stateful,
                   reverse=reverse,
                   return_state=return_state,
                   share_state_with=share_state_with,
                   name=name)

    def reset(self):
        if self.stateful:
            return tf.group([state.reset() for state in self.cell.previous_state])
        else:
            return None


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
                 activation=tf.tanh,
                 w_init=tf.initializers.glorot_uniform(),
                 u_init=tf.initializers.glorot_uniform(),
                 bias_init=tf.initializers.zeros(),
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
                         state_size=[n_units],
                         n_units=n_units,
                         activation=activation,
                         bias_init=bias_init,
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

        # tensor, state = self._build_graph()
        # self.compute()= tensor
        # self.layer_state =[state]

    def init_state(self):
        layer_state = super().init_state()
        input_layer = self.input_layers[0]
        previous_state = self.previous_state[0]

        with layer_scope(self):
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_state = self.r_reg(previous_state)

            if self.share_state_with is None:
                w = Linear(input_layer, self.n_units, add_bias=True, weight_init=self.w_init, name="w")
                u = Linear(previous_state, self.n_units, add_bias=False, weight_init=self.u_init, name="r_w")
            else:
                w = self.share_state_with.w
                u = self.share_state_with.u
                # this means the previous layer was regularized we want the inner layer
                # get inner state of dropconnect or other views
                if not self.regularized:
                    w = w.inner_layer if isinstance(w, ViewLayer) else w
                    u = u.inner_layer if isinstance(u, ViewLayer) else u

                w = w.reuse_with(input_layer)
                u = u.reuse_with(previous_state)

            if self.regularized:
                self.name
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    w = self.w_reg(w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    u = self.u_reg(u)

            layer_state.w = w
            layer_state.u = u

            output = Add(w, u)
            output = Activation(output, self.activation)

            state = Module(inputs=[input_layer, previous_state],
                           output=output,
                           name=self.name + "_h")

            if self.regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            output = Module(inputs=[input_layer, previous_state],
                            output=output,
                            name=self.name + "_output")

            self.output = output
            self.state = tuple([state])

        return layer_state

    def compute(self, input_layer, *previous_state):

        # I don't think we need to test this here, I should also remove any verifications from other compute functions
        # all verification should be done at input time, any layer that allows for dynamic inputs should document that
        # if previous_state and len(previous_state) != len(self.state_size):
        #     raise ValueError(f"previous state:\n"
        #                      f"\thas {len(previous_state)} elements\n"
        #                      f"\texpected {self.state_size}")

        # input_layer = self.input_layers[0] if input_layer is None else input_layer
        # previous_state = self.previous_state if previous_state is None else tuple(as_list(previous_state))

        # TODO problem with previous state when this might be a tuple
        # since this might be a tuple and we know how many inputs the layer has
        # we just need to get the previous state that is based on possible multiple arguments
        # the code is cleaner as well

        # previous_state = tuple(as_list(previous_state))
        output = self.output.compute(input_layer, *previous_state)

        return output


# TODO needs tests
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
            input_layer: a Layer to be gated
            gate_input: a layer to be used as the gate input
            gate_fn: function for gate
    """

    def __init__(self, input_layer, gate_input, gate_fn=tf.sigmoid, name="gate"):
        self.gate_fn = gate_fn
        super().__init__(input_layers=[input_layer, gate_input],
                         n_units=input_layer.n_units,
                         dtype=tf.float32,
                         name=name)

    def compute(self, input_tensor=None, gate_tensor=None):
        # input_layer = input_layer if input_layer is not None else self.input_layers[0]
        # gate_input = gate_input if gate_input is not None else self.input_layers[1]

        # input_tensor = as_layer(input_layer).compute()
        # gate_tensor = as_layer(gate_input).compute()

        with layer_scope(self):
            gate = self.gate_fn(gate_tensor)
            output = txf.apply_gate(input_tensor, gate)

            return output

    def reuse_with(self, input_layer, gate_input=None, name=None):
        if gate_input is None:
            gate_input = self.gate_input

        if name is None:
            name = self.name

        return Gate(input_layer=input_layer,
                    gate_input=gate_input,
                    gate_fn=self.gate_fn,
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
        embedding_shape: lookup table feature dimension
        batch_size: number of sequences to be looked up,
        if not None, will force a padding up to the specified batch_size
    """

    def __init__(self,
                 input_layer,
                 seq_size,
                 embedding_shape,
                 weight_init=tf.initializers.glorot_uniform(),
                 batch_size=None,
                 add_bias=False,
                 bias_init=tf.initializers.zeros(),
                 bias=None,
                 weights=None,
                 dtype=tf.float32,
                 name="lookup",
                 share_state_with=None,
                 batch_padding=True
                 ):

        self.weight_init = weight_init
        self.embedding_shape = embedding_shape
        self.seq_size = seq_size
        self.batch_padding = batch_padding

        self.add_bias = add_bias
        self.bias_init = bias_init
        self.bias = bias

        n_units = embedding_shape[-1]

        self.batch_size = batch_size

        self.weights = weights
        self.share_state_with = share_state_with

        super().__init__(input_layers=input_layer, n_units=n_units, dtype=dtype, name=name)

    def init_state(self):
        layer_state = super().init_state()

        # validate shared state
        if self.weights is not None:
            weight_shape = self.weights.get_shape().as_list()
            if self.embedding_shape != weight_shape:
                raise ValueError(
                    "shared weight shape {} and feature shape {} mismatch".format(weight_shape, self.embedding_shape))

        if self.bias is not None:
            num_bias = self.bias.get_shape().as_list()[-1]
            if self.embedding_shape[0] != num_bias:
                raise ValueError(
                    "number of bias {} and number of feature rows {} mismatch".format(num_bias,
                                                                                      self.embedding_shape[0]))

        if self.share_state_with is not None:
            if not isinstance(self.share_state_with, Lookup):
                raise TypeError("Layer can only share variables with other layer of the same type (Lookup)")

            if self.embedding_shape != self.share_state_with.embedding_shape:
                raise ValueError("Can only share variables with layers with the same feature shape: "
                                 "share_state_with is provided but \n"
                                 f"self shape: {self.embedding_shape} different from "
                                 f"other shape: {self.share_state_with.embedding_shape}")
        else:
            for dim in self.embedding_shape:
                if not isinstance(dim, int):
                    TypeError(f"Embedding shape should be a list of int: {type(dim)} found")

        with layer_scope(self):
            # init weights
            weights = self.share_state_with.weights if self.share_state_with is not None else self.weights
            if weights is None:
                init_value = self.weight_init(self.embedding_shape, dtype=self.dtype)
                weights = tf.Variable(initial_value=init_value,
                                      name="weights",
                                      trainable=True)

            layer_state.weights = weights

            bias = self.share_state_with.bias if self.share_state_with is not None else self.bias
            if self.add_bias:
                if bias is None:
                    bias = tf.Variable(initial_value=self.bias_init([self.embedding_shape[0]], self.dtype),
                                       name="bias", trainable=True)
            else:
                bias = None

            layer_state.bias = bias
        return layer_state

    def compute(self, input_tensor):
        input_tensor = as_tensor(input_tensor)
        if isinstance(input_tensor, tf.SparseTensor) and self.seq_size is None:
            raise ValueError("cannot use unknown seq_size with sparse inputs")

        if not isinstance(input_tensor, tf.SparseTensor) and input_tensor.dtype not in (tf.int32, tf.int64):
            # TODO each layers should prefix its errors with it's name?
            raise TypeError(f"invalid input layer dtype {input_tensor.dtype}: Lookup requires {tf.int32} or {tf.int64}")

        try:
            tf.assert_less(tf.rank(input_tensor), 3)
        except tf.errors.InvalidArgumentError:
            raise ValueError("expected 1D/2D input tensor")

        # TODO this validation fails if input seq_size is zero, it can happen
        if not isinstance(input_tensor, tf.SparseTensor):
            shape = tf.shape(input_tensor)
            if self.seq_size is not None:
                try:
                    tf.less_equal(shape[0], self.seq_size)
                except tf.errors.InvalidArgumentError:
                    raise ValueError("input layer n_units ({}) and seq_size ({}) should match for dense input layers \n"
                                     "if n_units < seq_size the lookup will be padded".format(shape[-1],
                                                                                              self.seq_size))

        with layer_scope(self):
            # batch size is unknown for sparse lookups
            # y = xW
            if isinstance(input_tensor, tf.SparseTensor):
                sp_dim = tf.cast(input_tensor.dense_shape[-1], tf.int32)

                # ops.py 1D sparse lookups into 2D sparse lookup with 3 lookups
                # similar to the semantics of 1D dense tensor lookups
                if len(input_tensor.get_shape().as_list()) == 1:
                    sp_batch_size = tf.shape(input_tensor.values)[0]
                    sp_indices = txf.matrix_indices(input_tensor.indices)
                    sp_batch_dim = tf.cast(tf.stack([sp_batch_size, sp_dim]), tf.int64)
                    input_tensor = tf.SparseTensor(sp_indices, input_tensor.values, sp_batch_dim)

                sp_values = input_tensor
                sp_indices = txf.sparse_indices(sp_values)

                # sums the lookups for the same row
                # TODO check if this code has corrected the sparse gradient problem or if I need to add my own version
                # in the math module
                lookup_weights = tf.nn.embedding_lookup_sparse(params=self.weights,
                                                               sp_ids=sp_indices,
                                                               sp_weights=sp_values,
                                                               combiner="sum")

                if self.bias is not None:
                    # lookup bias
                    lookup_bias = tf.nn.embedding_lookup_sparse(params=self.bias,
                                                                sp_ids=sp_indices,
                                                                sp_weights=sp_values,
                                                                combiner="sum")

                    lookup_bias = tf.expand_dims(lookup_bias, -1)

                    lookup_weights += lookup_bias

                output = lookup_weights

                # pad lookup if layer.tensor.dense_shape[0] is not a multiple of self.seq_size
                # this can happen if different lookups have a different number of indices
                lookup_batch = tf.shape(output)[0]
                expected_lookup_batch = tf.cast(
                    tf.math.ceil(lookup_batch / self.seq_size) * tf.cast(self.seq_size, dtype=tf.float64),
                    tf.int32)
                lookup_padding = expected_lookup_batch - lookup_batch

                # lookup_padding = sp_batch_size % self.seq_size
                lookup_padding = tf.stack([[0, lookup_padding], [0, 0]])
                output = tf.pad(output, lookup_padding)

                # dynamic batch size with sparse tensors
                # batch_size = tf.cast(tf.math.ceil(sp_batch_size / self.seq_size), tf.int32)
                # batch_size = Print(batch_size, [batch_size], message="")
                # tensor = tf.reshape(tensor, tf.stack([-1, self.seq_size, self.n_units]))

                output_shape = tf.stack([-1, self.seq_size, self.n_units])
                output = tf.reshape(output, output_shape)

                # padding
                padding = []
                if self.batch_padding and self.batch_size is not None:
                    batch_padding = tf.math.maximum(self.batch_size - tf.shape(output)[0], 0)
                    padding.append([0, batch_padding])
                else:
                    padding.append([0, 0])

                padding.append([0, 0])
                padding.append([0, 0])

                padding = tf.stack(padding)
                output = tf.pad(output, padding)
            else:
                # layer is dense
                shape = tf.shape(input_tensor)
                n_units = shape[-1]
                # if n_units is None:
                #     n_units = tf.shape(input_layer.tensor())[-1]

                # input_tensor = tf.reshape(input_layer.tensor, tf.stack([-1, n_units]))
                lookup_weights = tf.nn.embedding_lookup(params=self.weights,
                                                        ids=input_tensor)

                if self.bias is not None:
                    lookup_bias = tf.nn.embedding_lookup(params=self.bias,
                                                         ids=input_tensor)

                    lookup_bias = tf.expand_dims(lookup_bias, -1)
                    lookup_weights += lookup_bias

                batch_size = tf.shape(input_tensor)[0]
                lookup_shape = tf.stack([batch_size, -1, self.n_units])
                output = tf.reshape(lookup_weights, lookup_shape)

                # padding
                padding = []
                if self.batch_padding and self.batch_size is not None:
                    batch_padding = tf.math.maximum(self.batch_size - tf.shape(output)[0], 0)
                    padding.append([0, batch_padding])
                else:
                    padding.append([0, 0])

                # pad to seq_size if se_size is specified
                if self.seq_size is not None:
                    seq_padding = tf.math.maximum(self.seq_size - n_units, 0)
                    padding.append([0, seq_padding])
                else:
                    padding.append([0, 0])

                padding.append([0, 0])
                padding = tf.stack(padding)
                output = tf.pad(output, padding)

        return output

    def as_concat(self):
        """ concatenates the sequence produced by a lookup

        Returns:

        """

        def concat_fn(x):
            if self.seq_size is None:
                seq_size = tf.shape(self.input_layers[-1]())[-1]
            else:
                seq_size = self.seq_size

            new_shape = tf.stack([-1, self.n_units * seq_size])
            return tf.reshape(x, new_shape)

        if self.seq_size and self.n_units:
            n_units = self.seq_size * self.n_units
        else:
            n_units = None

        # TODO check if wrap layer can infer n_units
        return Wrap(self,
                    n_units=n_units,
                    wrap_fn=lambda current_layer: Lambda(current_layer, fn=concat_fn),
                    forward_attributes=["weights", "bias", "seq_size"],
                    name="concat")

    def permute_batch_time(self):
        return Wrap(wrapped_layer=self,
                    n_units=self.n_units,
                    wrap_fn=lambda current_layer: Lambda(current_layer, fn=lambda x: tf.transpose(x, [1, 0, 2])),
                    forward_attributes=["weights", "bias", "seq_size"],
                    name="permute_batch_time")

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
        share_state_with = self if self.share_state_with is None else self.share_state_with

        if name is None:
            name = self.name

        return Lookup(input_layer,
                      seq_size=self.seq_size,
                      embedding_shape=self.embedding_shape,
                      batch_size=self.batch_size,
                      weights=self.weights,
                      weight_init=None,
                      dtype=self.dtype,
                      name=name,
                      share_state_with=share_state_with,
                      batch_padding=self.batch_padding)


class SeqConcat(Layer):
    """ Concat 3D Layer representing a sequence of vectors

    !!! warning
        You cannot feed this layer to an non-dynamic layer like Linear without specifying seq_size,
        the reason being that some layers like Linear, require the input dimensions to initialize their
        state/weights, working with a dynamic sequence is only fine after this is passed through a lookup
        layer and `s * embedding_size` never changes throughout the computation
    """

    def __init__(self, input_seq, seq_size=None, time_major=False, name="seq_concat"):
        if input_seq.n_units is not None and seq_size is not None:
            if seq_size <= 0:
                raise ValueError(f"expected seq_size >0, got seq_size={seq_size}")
            n_units = input_seq.n_units * seq_size
        else:
            n_units = None

        super().__init__(input_layers=input_seq,
                         n_units=n_units,
                         name=name,
                         time_major=time_major,
                         seq_size=seq_size)

    def compute(self, input_tensor):
        with layer_scope(self):
            shape = tf.shape(input_tensor)
            n = shape[-1]

            if self.time_major:
                input_tensor = tf.transpose(input_tensor, [1, 0, 2])
                seq_size = shape[0]
            else:
                seq_size = shape[1]

            return tf.reshape(input_tensor, [-1, n * seq_size])


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

    def __init__(self, layer1, layer2, gate_input, gate_fn=tf.sigmoid, name="coupled_gate"):

        self.gate_fn = gate_fn
        self.gate_input = gate_input

        super().__init__(input_layers=[layer1, layer2, gate_input],
                         n_units=layer1.n_units,
                         dtype=tf.float32,
                         name=name)

    def compute(self, tensor1, tensor2, gate_input):
        with layer_scope(self):
            gate1 = self.gate_fn(gate_input)
            gate2 = 1 - gate1

            output1 = txf.apply_gate(tensor1, gate1)
            output2 = txf.apply_gate(tensor2, gate2)
            output = tf.math.add(output1, output2)

            return output

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


class GRUCell(BaseRNNCell):
    """ Gated Recurrent Unit Cell.

        Performs a single step with a gated recurrent unit where. These units have two gates:
        The first defines how much do we use the values from the recurrent connection to predict the current state
        The second
    """

    def __init__(self, input_layer, n_units,
                 previous_state=None,
                 activation=tf.tanh,
                 gate_activation=tf.sigmoid,
                 w_init=tf.initializers.glorot_uniform(),
                 u_init=tf.initializers.orthogonal(),
                 bias_init=tf.initializers.zeros(),
                 u_dropconnect=None,
                 w_dropconnect=None,
                 x_dropout=None,
                 r_dropout=None,
                 y_dropout=None,
                 dropout_locked=True,
                 regularized=False,
                 share_state_with=None,
                 name="gru_cell"):

        self.gate_activation = gate_activation
        self.output = None
        self.state = None

        super().__init__(input_layer=input_layer,
                         previous_state=previous_state,
                         state_size=None,
                         n_units=n_units,
                         activation=activation,
                         bias_init=bias_init,
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

    def init_state(self):
        layer_state = super().init_state()
        input_layer, previous_h = self.input_layers

        with layer_scope(self):
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_h = self.r_reg(previous_h)

            if self.share_state_with is None:
                # update gate kernel
                w_z = Linear(input_layer,
                             n_units=self.n_units,
                             add_bias=True,
                             bias_init=self.bias_init,
                             weight_init=self.w_init,
                             name="w_z")
                u_z = Linear(previous_h,
                             n_units=self.n_units,
                             weight_init=self.u_init,
                             add_bias=False,
                             name="u_z")

                # reset kernel
                w_r = Linear(input_layer,
                             n_units=self.n_units,
                             add_bias=True,
                             bias_init=self.bias_init,
                             weight_init=self.w_init,
                             name="w_r")
                u_r = Linear(previous_h,
                             n_units=self.n_units,
                             add_bias=False,
                             weight_init=self.u_init,
                             name="u_r")

                # output candidate kernel
                w_c = Linear(input_layer,
                             n_units=self.n_units,
                             add_bias=True,
                             bias_init=self.bias_init,
                             weight_init=self.w_init,
                             name="w_c")
                u_c = Linear(previous_h,
                             n_units=self.n_units,
                             add_bias=False,
                             bias_init=self.bias_init,
                             weight_init=self.u_init,
                             name="u_c")

                w = [w_z, w_r, w_c]
                u = [u_z, u_r, u_c]
            else:
                w = self.share_state_with.layer_state.w
                u = self.share_state_with.layer_state.u

                # in case the layer with which we share the state has a regularized state
                # TODO store both regularized ViewLayers in the state and normal layers
                # this way we can just retrieve the regularized state
                if not self.regularized:
                    w = list(map(lambda w: w.inner_layer if isinstance(w, ViewLayer) else w, w))
                    u = list(map(lambda u: u.inner_layer if isinstance(u, ViewLayer) else u, u))

                w = [wi.reuse_with(input_layer) for wi in w]
                u = [ui.reuse_with(previous_h) for ui in u]

                w_z, w_r, w_c = w
                u_z, u_r, u_c = u

            if self.regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    w = self.w_reg(*w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    u = self.u_reg(*u)

                w_z, w_r, w_c = w
                u_z, u_r, u_c = u

            layer_state.w_r = w_r
            layer_state.w_c = w_c
            layer_state.w_z = w_z
            layer_state.u_r = u_r
            layer_state.u_c = u_c
            layer_state.u_z = u_z
            layer_state.w = w
            layer_state.u = u

            r = Add(w_r, u_r, name="r")
            r_uc = Gate(u_c, r, name="gated_previous")
            candidate = Activation(Add(w_c, r_uc, name="candidate"), fn=self.activation, name="candidate")
            z = Add(w_z, u_z, name="z")
            # Note:
            #   (it's indifferent after training but) keras implementation is:
            #       h = z * prev_h + (1-z) * candidate
            #   and I had:
            #       h = z * candidate + (1-z) * prev_h
            #       (CoupledGate(candidate,previous_h z, name="output")
            #   but changed it to have comparable cells
            output = CoupledGate(previous_h, candidate, z, name="output")

            h = Module(inputs=[input_layer, previous_h],
                       output=output,
                       name=self.name + "_h")

            if self.regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            output = Module(inputs=[input_layer, previous_h],
                            output=output,
                            name=self.name + "_output")

            self.output = output
            self.state = tuple([h])

        return layer_state

    def compute(self, input_layer, *previous_state):
        # if previous_state and len(previous_state) != len(self.state_size):
        #     raise ValueError(f"previous state:\n"
        #                      f"\thas {len(previous_state)} elements\n"
        #                      f"\texpected {self.state_size}")

        # input_layer = self.input_layers[0] if input_layer is None else input_layer
        # previous_state = self.previous_state if previous_state is None else tuple(as_list(previous_state))

        output = self.output.compute(input_layer, *previous_state)
        return output

    def reuse_with(self, input_layer, *previous_state, regularized=None, name=None):
        return super().reuse_with(input_layer,
                                  *previous_state,
                                  regularized=regularized,
                                  name=name,
                                  gate_activation=self.gate_activation)


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
                 activation=tf.tanh,
                 gate_activation=tf.sigmoid,
                 bias_init=tf.initializers.zeros(),
                 forget_bias_init=tf.initializers.ones(),
                 w_init=tf.initializers.glorot_uniform(),
                 u_init=tf.initializers.glorot_uniform(),
                 w_dropconnect=None,
                 u_dropconnect=None,
                 x_dropout=None,
                 r_dropout=None,
                 y_dropout=None,
                 dropout_locked=True,
                 regularized=False,
                 share_state_with=None,
                 name="lstm_cell"):

        # attribute autocomplete
        self.w = None

        self.forget_bias_init = forget_bias_init
        self.gate_activation = gate_activation
        self.output = None
        self.state = None

        super().__init__(input_layer=input_layer,
                         previous_state=previous_state,
                         state_size=[n_units] * 2,
                         n_units=n_units,
                         activation=activation,
                         bias_init=bias_init,
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

    def init_state(self):
        layer_state = super().init_state()

        input_layer = self.input_layers[0]
        previous_h, previous_memory = self.previous_state

        with layer_scope(self):
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_h = self.r_reg(previous_h)

            # create new weights
            if self.share_state_with is None:

                # input gate linear
                w_i = Linear(input_layer, self.n_units, add_bias=True, bias_init=self.bias_init, name="w_i")
                u_i = Linear(previous_h, self.n_units, add_bias=False, name="u_i")

                # forget gate linear
                # http://proceedings.mlr.press/v37/jozefowicz15.pdf bias forget = 1
                forget_bias_init = self.bias_init if self.forget_bias_init is None else self.forget_bias_init
                w_f = Linear(input_layer, self.n_units, add_bias=True, bias_init=forget_bias_init, name="w_f")
                u_f = Linear(previous_h, self.n_units, add_bias=False, name="u_f")

                # candidate linear
                w_c = Linear(input_layer, self.n_units, add_bias=True, bias_init=self.bias_init, name="w_c")
                u_c = Linear(previous_h, self.n_units, add_bias=False, name="u_c")

                # output gate
                w_o = Linear(input_layer, self.n_units, add_bias=True, bias_init=self.bias_init, name="w_o")
                u_o = Linear(previous_h, self.n_units, add_bias=False, name="u_o")

                w = [w_i, w_f, w_c, w_o]
                u = [u_i, u_f, u_c, u_o]

            else:
                w = self.share_state_with.w
                u = self.share_state_with.u

                # get inner state of dropconnect or other views
                if not self.regularized:
                    w = list(map(lambda w: w.inner_layer if isinstance(w, ViewLayer) else w, w))
                    u = list(map(lambda u: u.inner_layer if isinstance(u, ViewLayer) else u, u))

                w = [wi.reuse_with(input_layer) for wi in w]
                u = [ui.reuse_with(previous_h) for ui in u]

                w_i, w_f, w_c, w_o = w
                u_i, u_f, u_c, u_o = u

            # apply regularizers to weights
            if self.regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    w = self.w_reg(*w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    u = self.u_reg(*u)

                w_i, w_f, w_c, w_o = w
                u_i, u_f, u_c, u_o = u

            layer_state.w_f = w_f
            layer_state.w_i = w_i
            layer_state.w_c = w_c
            layer_state.w_o = w_o
            layer_state.u_f = u_f
            layer_state.u_i = u_i
            layer_state.u_c = u_c
            layer_state.u_o = u_o
            layer_state.w = w
            layer_state.u = u

            with tf.name_scope("memory_forget"):
                gate_f = Add(w_f, u_f, name="add_f")
                memory_state = Gate(previous_memory, gate_f, gate_fn=self.gate_activation, name="gated_memory")

            with tf.name_scope("candidate_store"):
                gate_i = Add(w_i, u_i, name="candidate_gate")
                candidate = Activation(Add(w_c, u_c), fn=self.activation,
                                       name="candidate_activation")
                candidate = Gate(candidate, gate_i, gate_fn=self.gate_activation, name="gated_candidate")
                memory_state = Add(memory_state, candidate, name="add_to_memory")

                # wrap memory transformation with something that can be treated as a layer
                memory_state = Module(inputs=[input_layer, previous_h, previous_memory],
                                      output=memory_state,
                                      name=self.name + "_memory")

            with tf.name_scope("output"):
                gate_o = Add(w_o, u_o, name="add_o")
                output = Activation(memory_state, fn=self.activation, name="output")
                output = Gate(output, gate_o, gate_fn=self.gate_activation, name="gated_output")

            h = Module(inputs=[input_layer, previous_h, previous_memory],
                       output=output,
                       name=self.name + "_h")

            # when we use y_regularized we don't want to regularize the previous state
            # when we reuse cells, only the output
            if self.regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            output = Module(inputs=[input_layer, previous_h, previous_memory],
                            output=output,
                            name=self.name + "_output")

        self.output = output
        self.state = (h, memory_state)
        return layer_state

    def compute(self, input_tensor, *previous_state):
        """ compute layer value based on input `Tensor` values

        Args:
            input_tensor: a `Tensor` or `Layer` input to the current cell
            *previous_state: (previous_h, previous_memory)

        Returns:
            `Constant`: a tensor with the cell's output

        """
        previous_h, previous_memory = previous_state
        output = self.output.compute(input_tensor, previous_h, previous_memory)
        return output

    def reuse_with(self, input_layer, *previous_state, regularized=None, name=None):
        # TODO change reuse_with with input_layer, *previous_state
        return super().reuse_with(input_layer,
                                  *previous_state,
                                  regularized=regularized,
                                  name=name,
                                  gate_activation=self.gate_activation,
                                  forget_bias_init=self.forget_bias_init)


class Activation(Layer):
    """Activation(layer,fn=tf.identity,name="activation",**kwargs)

        Applies a given function the the output of its input layer.

        You can pass positional arguments and keyword arguments for the given function,
        their application works like :func:`functools.partial`.


        Warnings:
            if the input layer outputs a ``SparseTensor``, this is converted to a dense ``Tensor`` first.

        Args:
            input_layer: the input :class:`Layer`
            fn: a function that produces a Tensor and can be called on the tensor produced by the input layer
            name: the layer name
            **kwargs: the keyword arguments for the given function

    """

    def __init__(self, input_layer, fn=tf.identity, name="activation", **kwargs):
        self.fn = partial(fn, **kwargs)
        self.kw = kwargs
        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=input_layer.dtype,
                         name=name)

    def compute(self, input_tensor):
        with layer_scope(self):
            if isinstance(input_tensor, tf.SparseTensor):
                input_tensor = tf.sparse.to_dense(input_tensor)
            output = self.fn(input_tensor)
            return output

    def reuse_with(self, layer, name=None):
        name = self.name if name is None else name
        return Activation(layer, self.fn, name, **self.kw)


class Merge(Layer):
    """Merge Layer

    Merges a list layers by combining their tensors with a merging function.
    Allows for the tensor of each layer to be weighted.

    This is just a container that for convenience takes the tensor of each given layer (which is generaly a tensor),
    and applies a merging function.

    Args:
            input_layers: a list of layers with the same number of units to be merged
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
                 *input_layers,
                 n_units=None,
                 dtype=None,
                 weights=None,
                 merge_fn=tf.math.add_n,
                 name="merge",
                 **kwargs):

        if len(input_layers) < 1:
            raise Exception("You must provide at least one layer")

        if weights is not None and len(weights) != len(input_layers):
            raise Exception("len(weights) must be equals to len(layers)")

        self.weights = weights
        self.merge_fn = merge_fn

        super().__init__(input_layers=input_layers, n_units=n_units, dtype=dtype, name=name, **kwargs)

    def compute(self, *input_tensors):
        outputs = [as_tensor(ts) for ts in input_tensors]
        if self.weights is not None:
            outputs = [tf.math.scalar_mul(self.weights[i], outputs[i]) for i in
                       range(len(outputs))]

        with layer_scope(self):
            output = self.merge_fn(outputs)

            # shape = tf.shape(output)

            # if self.n_units is None:
            #     if len(shape) > 0:
            #         self.n_units = shape[-1]
            # elif self.n_units > 0:
            #     if shape[-1] != self.n_units:
            #         raise ValueError(
            #             f"output n_units {tf.shape(output)[-1]} does not match n_units {self.n_units}")

            if self.dtype is None:
                self.dtype = output.dtype
            else:
                if self.dtype != output.dtype:
                    output = tf.cast(output, self.dtype)
            output = tf.cast(output, self.dtype)

        return output

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name
        return Merge(*layers, weights=self.weights, merge_fn=self.merge_fn, name=name)


class Add(Merge):
    """ Adds the outputs of multiple layers with the same shape

    Args:
            input_layers: a list of layers with the same number of units to be merged
            weights: a list of weights
            name: name for layer scope
    """

    def __init__(self, *input_layers, weights=None, name="add"):
        input_layers = list(map(lambda l: as_layer(l), input_layers))

        n_units = input_layers[0].n_units
        dtype = input_layers[0].dtype
        for lr in input_layers[1:]:
            if lr.n_units is not None:
                if lr.n_units != n_units:
                    raise ValueError("Found layers with different sizes of n_units {}!={} in an Add Layer".format(
                        n_units,
                        lr.n_units
                    ))
            if lr.dtype is not None:
                if lr.dtype != dtype:
                    raise ValueError("Found layers with different dtypes {}!={} in an Add Layer".format(
                        dtype,
                        lr.dtype
                    ))

        def merge_add(tensors):
            res = 0
            for tensor in tensors:
                res = res + tensor
            return res

        # merge_add = tf.add_n

        super().__init__(*input_layers, n_units=n_units, dtype=dtype, weights=weights, merge_fn=merge_add, name=name)


class Concat(Merge):
    """ Concat Layer

    Concatenates input layers on the last dimension

    Args:
        input_layers: a :obj:`list` of :class:`Layer`
        name: name for the layer scope
    """

    def __init__(self, *input_layers, axis=-1, name="concat"):
        input_layers = list(map(lambda lr: as_layer(lr) if not isinstance(lr, Layer) else lr, input_layers))
        first, *rest = input_layers
        if not all(lr.dtype == first.dtype for lr in rest):
            raise ValueError("Layers must have the same type to be concatenated")

        n_units = sum([lr.n_units for lr in input_layers])
        super().__init__(*input_layers,
                         n_units=n_units,
                         merge_fn=partial(tf.concat, axis=axis),
                         dtype=first.dtype,
                         name=name,
                         axis=axis)


class Residual(Layer):
    """ Residual Block
    """

    def __init__(self,
                 x_layer,
                 h_layer,
                 share_state_with=None,
                 weight_init=tf.initializers.glorot_uniform(),
                 name="residual"):

        if share_state_with is not None:
            if not isinstance(share_state_with, Residual):
                raise TypeError("can only share vars with a Residual Layer {} found".format(type(share_state_with)))

        # try to create a module from the x_layer -> h_layer
        # if one is not connected to the other, this fails
        self.module = Module(x_layer, h_layer)

        super().__init__(input_layers=[x_layer, h_layer],
                         n_units=h_layer.n_units,
                         dtype=h_layer.dtype,
                         name=name,
                         share_state_with=share_state_with,
                         weight_init=weight_init)

    def init_state(self):
        x, h = self.input_layers
        state = super().init_state()
        with layer_scope(self):
            if x.n_units != h.n_units:
                if self.share_state_with is None:
                    state.projection = Linear(x, h.n_units, weight_init=self.weight_init, add_bias=False)
                else:
                    state.projection = self.share_state_with.projection.reuse_with(x)
                output = Add(h, state.projection)
            else:
                output = Add(h, x)

            self.output = Module(inputs=[x, h], output=output, name="residual_block")
        return state

    def compute(self, x, h):
        return self.output.compute(x, h)

    def reuse_with(self, x_layer, h_layer, name=None):
        if name is None:
            name = self.name

        share_state_with = self if self.share_state_with is None else self.share_state_with

        return Residual(x_layer=x_layer,
                        h_layer=h_layer,
                        share_state_with=share_state_with,
                        name=name)


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

    Assumes the input to have a shape (batch,time_step,n)
    produces an output of shape (batch,time_step,m) where m is the number of filters

    Args:
        input_layer: input `Layer` with shape (batch,time_step,dim)
        n_units: number of output units for this layer (number of filters)
        filter_size: convolution filter size
    """

    def __init__(self, input_layer,
                 n_units,
                 filter_size,
                 stride=1,
                 dilation_rate=1,
                 same_padding=True,
                 filter_init=tf.initializers.glorot_uniform(),
                 bias_init=tf.initializers.zeros(),
                 add_bias=True,
                 name="conv1D",
                 share_state_with=None,
                 filters=None,
                 bias=None):

        # TODO should shared bias and shared weights/filters etc be in the proto? I guess pickling this requires
        #   the storage of the respective weights
        # self.shared_bias = shared_bias
        # self.share_state_with = share_state_with
        # self.shared_filters = shared_filters
        self.padding = "SAME" if same_padding else "VALID"
        self.same_padding = same_padding
        self.dilation_rate = dilation_rate
        self.stride = stride
        self.filter_size = filter_size
        self.filter_init = filter_init
        self.bias_init = bias_init
        self.add_bias = add_bias
        self.bias = bias
        self.filters = filters
        self.share_state_with = share_state_with

        # input_tensor_shape = input_layer.tensor.get_shape()
        # output_shape = _conv_out_shape(input_tensor_shape, self.filter_shape, self.padding, stride, dilation_rate)
        # self.output_shape = tf.TensorShape(output_shape).as_list()

        super().__init__(input_layers=input_layer,
                         n_units=n_units,
                         dtype=tf.float32,
                         name=name
                         )

    def init_state(self):
        input_layer = self.input_layers[0]
        filter_shape = [self.filter_size, input_layer.n_units, self.n_units]

        if self.share_state_with is not None:
            if not isinstance(self.share_state_with, Conv1D):
                raise TypeError("Layer can only share variables with other layer of the same type")

            layer_state = self.share_state_with.layer_state
        else:
            layer_state = super().init_state()

        with layer_scope(self):
            filters = getattr(layer_state, "filters", self.filters)

            if filters is None:
                init_value = self.filter_init(filter_shape, dtype=self.dtype)
                filters = tf.Variable(initial_value=init_value,
                                      dtype=self.dtype,
                                      name="filters")

            if not hasattr(layer_state, "filters"):
                layer_state.filters = filters

            bias = getattr(layer_state, "bias", self.bias)
            if self.add_bias:
                if bias is None:
                    bias = tf.Variable(initial_value=self.bias_init([self.n_units], self.dtype),
                                       name="bias", trainable=True)

            if not hasattr(self.layer_state, "bias"):
                self.layer_state.bias = bias

        return layer_state

    def compute(self, input_tensor):
        with layer_scope(self):
            if isinstance(input_tensor, tf.SparseTensor):
                input_tensor = tf.sparse.to_dense(layer.tensor)

            # if input_tensor.dtype == tf.float64:
            #     input_tensor = tf.cast(input_tensor, tf.float32)

            output = tf.nn.convolution(input=input_tensor,
                                       filters=self.layer_state.filters,
                                       padding=self.padding,
                                       strides=(self.stride,),
                                       dilations=(self.dilation_rate,),
                                       data_format="NWC")

            if self.add_bias:
                output = tf.nn.bias_add(output, self.layer_state.bias, name="add_b")

            return output

    def reuse_with(self, input_layer, name=None):
        share_state_with = self if self.share_state_with is None else self.share_state_with
        if name is None:
            name = self.name

        return Conv1D(input_layer=input_layer,
                      n_units=self.n_units,
                      filter_size=self.filter_size,
                      stride=self.stride,
                      dilation_rate=self.dilation_rate,
                      same_padding=self.same_padding,
                      filter_init=self.weight_init,
                      bias_init=self.bias_init,
                      add_bias=self.add_bias,
                      name=name,
                      bias=self.bias,
                      filters=self.filters,
                      share_state_with=share_state_with)


class MHAttention(Layer):
    """ Scaled Dot Product MultiHead Attention Layer

    (Q,K,V):
        Encodes representation of the input as a set of key-value pairs, (K,V), both of dimension n (input sequence
        length); in the context of sequence-to-sequence models, both keys and values are the encoder hidden states.
        In the decoder, the previous output is compressed into a query (Q of dimension m)

    Args:
        query:
        key:
        value:
        n_units: output number of units, each attentio head has n_units // n_head units, meaning that n_units needs to
        be a multiple of n_heads.

    """

    def __init__(self,
                 query,
                 key,
                 value,
                 n_units=None,
                 n_heads=1,
                 attention_fn=tf.nn.softmax,
                 causality=False,
                 attention_dropout=0.0,
                 regularized=False,
                 name="attention",
                 share_state_with=None):
        self.n_heads = n_heads
        n_units = query.n_units if n_units is None else n_units
        self.causality = causality
        self.share_state_with = share_state_with
        self.regularized = regularized
        self.attention_dropout = attention_dropout
        self.attention_fn = attention_fn

        if n_units % n_heads != 0:
            raise ValueError(
                "The n_units {} is not a multiple of the number of attention "
                "heads {}".format(self.n_units, n_heads))

        self.head_units = n_units // n_heads
        super().__init__(input_layers=[query, key, value], n_units=n_units, name=name)

    def init_state(self):
        if self.share_state_with is not None:
            if not isinstance(self.share_state_with, MHAttention):
                raise TypeError("Layer can only share state with other layer of the same type")

            layer_state = self.share_state_with.layer_state

        else:
            layer_state = super().init_state()

        query, key, value = self.input_layers
        h_dim = self.n_units

        with layer_scope(self):
            if self.share_state_with is None:
                # (batch_size, t, n_units)
                wq = Linear(query, n_units=h_dim, add_bias=False, name="wq")
                wk = Linear(key, n_units=h_dim, add_bias=False, name="wk")
                wv = Linear(value, n_units=h_dim, add_bias=False, name="wv")

                layer_state.wq = wq
                layer_state.wk = wk
                layer_state.wv = wv

        return layer_state

    def compute(self, *input_tensors):
        query, key, value = input_tensors

        # (n_heads*batch_size, steps, n_units//n_heads)
        def heads(w):
            return tf.concat(tf.split(w, self.n_heads, axis=2), axis=0)

        with layer_scope(self):
            dk = self.n_units
            wq = self.wq.compute(query)
            wk = self.wk.compute(key)
            wv = self.wv.compute(value)

            qh = heads(wq)
            kh = heads(wk)
            vh = heads(wv)

            # attention scores from scaled dot product
            dotprod = tf.matmul(qh, tf.transpose(kh, [0, 2, 1]))

            # hypothesis: for large values of √dk, the dot products grow large in magnitude, pushing the
            # softmax function into regions with extremely small gradients. To counteract this effect, we scale
            # the dot products by1 √dk.
            dotprod /= dk ** 0.5
            output = dotprod

            # mask information from the future
            if self.causality:
                diag_values = tf.ones_like(output[0, :, :])  # (tq, tk)
                triangular = tf.linalg.LinearOperatorLowerTriangular(diag_values).to_dense()  # (tq, tk)
                masks = tf.tile(tf.expand_dims(triangular, 0), [tf.shape(output)[0], 1, 1])  # (N, tq, tk)
                # mask to - inf before softmax
                padding = tf.ones_like(masks) * (-2 ** 32 + 1)
                output = tf.where(tf.equal(masks, 0), padding, output)

            scores = self.attention_fn(output)

            if self.attention_dropout > 0 and self.regularized:
                scores = txf.dropout(tensor=scores,
                                     probability=self.attention_dropout,
                                     scale=True,
                                     name="dropout")

            # weighted sum (context vectors) weighted by attention scores
            context_vectors = tf.matmul(scores, vh)
            # restore shape (batch_size, tq, n_units)
            output = tf.concat(tf.split(context_vectors, self.n_heads, axis=0), axis=2)

            return output

    def reuse_with(self, query, key, value, regularized=None, causality=None, name=None):
        regularized = self.regularized if regularized is None else regularized
        name = self.name if name is None else name
        causality = self.causality if causality is None else causality

        return MHAttention(query=query,
                           key=key,
                           value=value,
                           n_units=self.n_units,
                           attention_fn=self.attention_fn,
                           causality=causality,
                           n_heads=self.n_heads,
                           attention_dropout=self.attention_dropout,
                           regularized=regularized,
                           name=name,
                           share_state_with=self)


class FC(Layer):
    def __init__(self,
                 input_layer,
                 n_units,
                 activation=tf.identity,
                 weight_init=tf.initializers.glorot_uniform(),
                 weights=None,
                 transpose_weights=False,
                 add_bias=True,
                 bias_init=tf.initializers.zeros(),
                 bias=None,
                 weight_norm=False,
                 dtype=tf.float32,
                 name="fc",
                 share_state_with=None):

        # for attribute autocomplete only
        self.linear = None
        self.activation = None

        super().__init__(input_layers=input_layer,
                         n_units=n_units,
                         dtype=dtype,
                         name=name,
                         activation=activation,
                         weight_init=weight_init,
                         weights=weights,
                         transpose_weights=transpose_weights,
                         add_bias=add_bias,
                         bias_init=bias_init,
                         bias=bias,
                         weight_norm=weight_norm,
                         share_state_with=share_state_with
                         )

    def init_state(self):
        input_layer = self.input_layers[0]
        state = super().init_state()

        with layer_scope(self, name=self.name):
            if self.share_state_with is None:
                linear = Linear(input_layer=input_layer,
                                n_units=self.n_units,
                                weight_init=self.weight_init,
                                weights=self.weights,
                                transpose_weights=self.transpose_weights,
                                add_bias=self.add_bias,
                                bias_init=self.bias_init,
                                bias=self.bias,
                                dtype=self.dtype,
                                weight_norm=self.weight_norm,
                                name="{}_linear".format(self.name),
                                share_state_with=self.share_state_with)
            else:
                linear = self.share_state_with.linear.reuse_with(input_layer)

            activation = Activation(linear, fn=self.activation, name="{}_activation".format(self.name))
            state.linear = linear
            state.activation = activation

            self.output = Module(inputs=input_layer, output=activation)

        return state

    def compute(self, input_tensor):
        return self.output.compute(input_tensor)


# TODO careful with reuse_with was not useing share_state_with=self
#    def reuse_with(self, input_layer):
#        return self.proto(input_layer, share_state_with=self)


class SeqMap(Layer):
    """ Applies a given layer prototype to each element in the first dimension of the input layer

    """

    def __init__(self,
                 input_seq,
                 layer_proto: Callable[[Union[Layer, tf.Tensor]], Layer],
                 parallel_iterations=10,
                 n_units=None,
                 share_state_with: Optional['SeqMap'] = None,
                 name="seq_map"):

        expected_n = layer_proto.arg_dict.get('n_units', None)
        if expected_n is not None and n_units is not None and n_units != expected_n:
            raise ValueError(f"n_units of layer instance does not match expected n_units:\n"
                             f"\t expected: {expected_n}\n"
                             f"\t got: {n_units}")

        # n_units and shape are set after the first cell is created
        super().__init__(input_layers=input_seq,
                         n_units=layer_proto.arg_dict.get('n_units', n_units),
                         dtype=tf.float32,
                         name=name,
                         layer_proto=layer_proto,
                         parallel_iterations=parallel_iterations,
                         share_state_with=share_state_with)

    def init_state(self):
        state = super().init_state()
        input_seq = self.input_layers[-1]
        x0 = input_seq[0]

        # TODO we could pass the layer directly but since right now we only allow to build layers from
        # a given input layer, we can just past the layer proto, the specification of the layer to be applied
        # an alternative is to allow layers to be created based on an input shape
        # I can generalize this to all layers and nothing changes besides that, every layer would have an input shape
        # and if provided it would be used to initialize its state
        #
        # There would be two ways of specifying layers, using a layer proto object and a layer object with
        # an input_shape, layer proto would be a way to delay the initialization of the layer state, layer with input shape
        # would allow me to build the state immediately
        if self.share_state_with is not None:
            layer_instance = self.share_shate_with.layer_instance
            state.layer_instance = layer_instance.reuse_with(x0)
        else:
            state.layer_instance = self.layer_proto(x0)

        n_units = state.layer_instance.n_units
        if n_units is not None and self.n_units is None:
            self.n_units = state.layer_instance.n_units
        elif n_units is not None and self.n_units != n_units:
            raise ValueError(f"n_units of layer instance does not match expected n_units:\n"
                             f"\t expected: {self.n_units}\n"
                             f"\t got: {n_units}")

        return state

    def compute(self, input_seq):
        layer_instance = self.layer_instance
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

            output_ta = output_ta.write(i0, layer_instance.compute(x0))

            def compute_step(t, y):
                xt = input_ta.read(t)
                c = layer_instance.compute(xt)
                y = y.write(t, c)
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
                      parallel_iterations=self.parallel_iterations,
                      share_state_with=self,
                      name=name)


def as_layer(layer_like: Union[tf.Tensor, Layer], dtype=None):
    """ Converts a ``Tensor``,``SparseTensor`` or tensor convertible to a ``Layer``

    Args:
        dtype: if not None and different from the input dtype, tries to cast the output layer to the given dtype
        layer_like: a layer, tensor, or convertible to tensor

    Returns:
        the input ``Layer`` or, a ``Layer`` with the given value
    """
    if isinstance(layer_like, Layer):
        if dtype is not None and layer_like.dtype != dtype:
            layer_like = Lambda(layer_like, lambda x: tf.cast(x, dtype=dtype), apply_to_layer=False)
        return layer_like
    else:
        tensor = as_tensor(layer_like, dtype)
        return Constant(tensor, dtype=dtype)


# register Layer to tensor conversion
def layer_to_tensor(input_layer, dtype=None, name=None, as_ref=False):
    name = name if name is not None else input_layer.name
    with tf.name_scope(name):
        return as_tensor(input_layer(), dtype=dtype)


tf.register_tensor_conversion_function(
    base_type=Layer,
    conversion_func=layer_to_tensor,
    priority=100
)


def layer(n_units=None, name="layer", dtype=None, var_list=None):
    """ Decorator for functions that returns a layer prototype

    Returns:
        proto (`LayerProto`): instance that can be called on layers to create a new layer instance
    """

    def function_to_proto(fn):
        if isinstance(fn, LayerProto):
            return fn
        return Lambda.proto(fn=fn, n_units=n_units, dtype=dtype, var_list=var_list, name=name)

    return function_to_proto


__all__ = [
    "DropConnect",
    "as_layer",
    "Activation",
    "Lambda",
    "Layer",
    "layer",
    "Linear",
    "Input",
    "Constant",
    "Param",
    "Wrap",
    "VariableLayer",
    "Transpose",
    "Reshape",
    "Add",
    "Concat",
    "Module",
    "Gate",
    "CoupledGate",
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    "RNN",
    "Lookup",
    "ToDense",
    "ToSparse",
    "Dropout",
    "Conv1D",
    "MHAttention",
    "DropLookup",
    "Residual",
    "FC",
    "SeqConcat",
    "SeqMap",
    "LayerNorm"
]
