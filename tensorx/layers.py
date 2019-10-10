import tensorflow as tf
from tensorx.utils import as_tensor, as_list, Graph
from typing import List, Union, Type, Callable, Optional, Iterable
import inspect
from contextlib import ExitStack
import tensorx.transform as txf
from collections import deque
from tensorx.callbacks import OnValueChange
from functools import partial
from tensorx.math import embedding_lookup_sparse


class LayerState:
    """ Layer state is used as a namespace to store either ``tf.Variable`` or ``Layer`` instances
    that contain ``tf.Variables`` that define the state of a ``Layer``.

    Notes:
        ideally ``LayerState`` objects would contain only ``Variable`` instances, but
        storing Layers can be more convenient.
    """
    pass

    def variables(self):
        """ returns a list of all variables contained in the layer state object

        Returns:
            list: a list of all variables in the layer state

        """
        return list(self.var_dict().values())

    def var_dict(self):
        """ returns a dictionary where the keys are the attribute names
        in a layer state or the attribute names in an inner layer of the layer state.

        Returns:
            dict: a dictionary from attribute names to variable instances with all
            the variables in the current layer state

        """
        all_vars = dict()
        for attr, state in self.__dict__.items():
            if isinstance(state, Layer):
                v = state.layer_state.var_dict()
                all_vars.update(v)
            elif isinstance(state, tf.Variable):
                all_vars[attr] = state
        return all_vars

    def __str__(self):
        return "State{\n %s \n}" % ("\n".join(["\t{}: {}".format(k, str(i)) for k, i in self.__dict__.items()]))


class LayerScope:
    """ LayerScope creates a unique name for the scope if not executing eagerly

    Args:
        layer: layer to be used in this scope, the layer name is used as scope name for tensorflow tf.name_scope
        and variable_scope, also modifies the layer name if the scope name clashes with existing names

        reuse: if True does not change the input layer name but it does create a unique name for tf.name_scope
        (for debug purposes only)
    """

    def __init__(self, layer, name=None, reuse=False):
        self.reuse = reuse
        self.layer = layer
        if name is not None and layer is not None:
            self.layer.name = name

        if layer is None and name is None:
            raise ValueError("layer scope needs a Layer or a name but both are None")

        self.name = self.layer.name if layer is not None else name
        self._stack = None

    def __enter__(self):
        with ExitStack() as stack:
            layer_name_scope = tf.name_scope(self.name)
            scoped_name = stack.enter_context(layer_name_scope)
            scoped_name = scoped_name[:-1]
            unique_unscoped_name = scoped_name[scoped_name.find(self.name):]

            if not self.reuse:
                self.name = unique_unscoped_name
                if self.layer is not None:
                    self.layer.name = self.name
                    self.layer.scoped_name = scoped_name
            else:
                scoped_name = self.name
            self._stack = stack.pop_all()
            return scoped_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)


layer_scope: Type[LayerScope] = LayerScope


class LayerProto:
    """ Layer Proto

    Creates a Layer prototype. A callable that allows us to delay calling the constructor a Layer object and validates
    constructor arguments.
    """

    def _validate_args(self, **kwargs):
        for key in kwargs:
            if key not in self.args:
                raise TypeError("{} prototype got an unexpected argument {}".format(self.layer_cls.__name__, key))

    def __init__(self, layer_cls, **kwargs):
        self.layer_cls = layer_cls
        spec = inspect.getfullargspec(layer_cls.__init__)
        self.args = set(spec.args[1:] + spec.kwonlyargs)
        self._validate_args(**kwargs)

        self.args_set = kwargs

    def __call__(self, *args, **kwargs):
        new_args = dict(self.args_set)
        new_args.update(kwargs)
        return self.layer_cls(*args, **new_args)

    def update(self, **kwargs):
        """ update.

        Updates the prototype constructor argument dictionary and validates those parameters.

        Args:
            **kwargs: new values for constructor named arguments to be updated
        """
        self._validate_args(**kwargs)
        self.args_set.update(kwargs)


class Layer:
    """ Layer base class

    Attributes:
        input_layers: a list of input nodes for the current layer
        n_units: the number of units for the current layer (last dim)
        name: name to be used for the layer scope
        scoped_name: layer full scope name
        variables: a list of `tf.Variable` that define the state of this layer

    Notes:
        * If a layer is created inside the scope of another layer, its scoped_name is the final name attributed
        to the layer taking the outer scopes into account.

    Args:
        input_layers: a single layer,a list of input layers, or None if no inputs are required
        n_units: dimension of input vector (dimension of columns in case batch_size != None
        name: layer name (used to nam the placeholder)

    """

    def __init__(self, input_layers, n_units, dtype=None, name="layer"):
        self.n_units = n_units
        self.name = getattr(self, "name", name)
        self.scoped_name = name
        self._input_layers = [as_layer(input_layer) for input_layer in as_list(input_layers)]
        self.dtype = tf.dtypes.as_dtype(dtype) if dtype is not None else None

        self.attr_names = []

        self.layer_state = None
        self.init_state()

        # TODO I think referring to state explicitly should be the preferred way to
        #   access it but this is only a problem if a shared state changes somehow
        #   e.g. two linear layers one has bias the second does not, if we add a bias
        #       to one that shares a state, it has to reuse the bias from the second layer
        # forward attributes from state to avoid layer.layer_state.variable
        if self.layer_state is not None:
            self.__dict__.update(self.layer_state.__dict__)

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
        return type(self)(*layers, **kwargs)

    @property
    def input_layers(self):
        return list(self._input_layers)

    @input_layers.setter
    def input_layers(self, input_layers):
        raise ValueError("input_layers can't be set")

    def compute(self, *args):
        raise NotImplementedError("computation not implemented for this layer")

    def compile_graph(self, input_signature=None):
        """
        Notes:
            I generally don't use python objects to control how the graph is created (e.g. number of layers)
            but if we use python objects we should convert them to tensors to make sure the graph is not
            retraced each time if the parameters do not affect the resulting graph.

            This is generally not needed because neural network layers do not have multiple traces because layers
            don't change types.

            double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))


        Reference:
            https://www.tensorflow.org/beta/tutorials/eager/tf_function
        Args:
            input_signature:

        Returns:

        """
        # @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))

        return tf.function(self.compute, input_signature)

    def init_state(self):
        self.layer_state = LayerState()
        return self.layer_state

    def __str__(self):
        """ Informal string representation for a layer consists of Layer Class name, number of units and if its
        Sparse or Dense.

        Returns:
            :obj:`str`: a :obj:`str` with the informal representation for a layer instance.

        """
        class_name = type(self).__name__
        return "{layer_name}::{class_name}({n_units},{dtype})({inputs})".format(layer_name=self.scoped_name,
                                                                                class_name=class_name,
                                                                                n_units=self.n_units,
                                                                                dtype=self.dtype,
                                                                                inputs=",".join(map(lambda x: x.name,
                                                                                                    self.input_layers))
                                                                                )

    def __getitem__(self, item):
        if isinstance(item, tf.Tensor):
            item_name = item.op.name
        else:
            item_name = str(item)
        return WrapLayer(layer=self,
                         n_units=self.n_units,
                         fn=lambda tensor: tensor[item],
                         name="{}_item_{}".format(self.name, item_name))

    def __call__(self, *input_layers):
        return self.compute(*input_layers)

    def tensor(self):
        """ tensor() alias for compute without arguments

        Returns:
            the results of the graph at the current layer
        """
        return self.compute()


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

        self.fn = fn
        self.var_list = as_list(var_list)
        self.apply_to_layer = apply_to_layer

        super().__init__(input_layers=layers,
                         n_units=n_units,
                         dtype=dtype,
                         name=name)

    def init_state(self):
        layer_state = super().init_state()
        for i, var in enumerate(self.var_list):
            setattr(layer_state, f"var_{i}", var)
        return layer_state

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_layers = [as_layer(x) for x in input_layers]
        inputs = [x.compute() if not self.apply_to_layer else x for x in input_layers]

        with layer_scope(self):
            output = self.fn(*inputs)
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
        super().__init__(input_layers=input_layer, n_units=input_layer.n_units, dtype=input_layer.dtype,
                         name=input_layer.name + "_sparse")

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        # there's no guarantee that input_layers have layer objects
        input_layers = [as_layer(x) for x in input_layers]

        input_layer = input_layers[0]
        input_value = input_layer.compute()

        with layer_scope(self):
            if isinstance(input_value, tf.SparseTensor):
                return input_value
            else:
                return txf.to_sparse(input_value)

    def reuse_with(self, input_layer):
        return ToSparse(input_layer)


class ToDense(Layer):
    """ ToDense transformation layer

    Transforms the previous layer into a dense layer (outputting a dense tensor)
    if the previous layer is already a dense layer, forwards the previous layer doing nothing

    """

    def __init__(self, input_layer):
        super().__init__(input_layers=input_layer, n_units=input_layer.n_units, dtype=input_layer.dtype,
                         name=input_layer.name + "_dense")

    # TODO should I return a list if it receives a list?
    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        # there's no guarantee that input_layers have layer objects
        input_layers = [as_layer(x) for x in input_layers]

        input_layer = input_layers[0]
        input_value = input_layer.compute()

        with layer_scope(self):
            if isinstance(input_value, tf.SparseTensor):
                return tf.sparse.to_dense(input_value)
            else:
                return input_value

    def reuse_with(self, input_layer):
        return ToDense(input_layer)


class WrapLayer(Layer):
    """ Wraps another layer with tf code

    Utility layer used to wrap arbitrary layers with another tensorflow graph op
    this might be useful to customize existing layers without creating a new layer from scratch


    Example::

    You can create nested WrapLayers in which case, ``reuse_with`` will replace the inputs
    of the innermost `Layer` (which is not a `WrapLayer`)

    .. aafig::
        :aspect: 60
        :scale: 150
        :proportional:
        :textual:

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
        fn
        apply_to_layer


    Args:
        layer: a `Layer` to be wrapped by this Layer
        n_units: the new number of units this layer will have
        fn: a callable returning a `Tensor` or `SparseTensor`
        name: name for this layer, defaults to wrap_[layer]
        apply_to_layer: if False applies the fn to the layer tensor outputs
        if false applies to the layer itself. if False applies to the layer itself and expects
        the output to be a tensor.



    """

    def __init__(self, layer, fn, n_units=None, forward_attributes=None, name="wrap", apply_to_layer=False):
        self.apply_to_layer = apply_to_layer
        self.fn = fn
        self.layer = layer
        self.n_units = n_units
        self.forward_attributes = as_list(forward_attributes)

        for attr in self.forward_attributes:
            if hasattr(layer, attr):
                setattr(self, attr, getattr(layer, attr))

        if name == "wrap":
            name = "wrap_{}".format(layer.name)

        super().__init__(input_layers=layer, n_units=n_units, dtype=layer.dtype, name=name)

    @property
    def variables(self):
        return self.layer.variables

    def compute(self, *input_layers):
        wrapped_layer = self.input_layers[0]

        if input_layers:
            input_layers = [as_layer(x) for x in input_layers]
            wrapped_layer = wrapped_layer.reuse_with(*input_layers)

        with layer_scope(self, name=self.name):
            fn_inputs = wrapped_layer.tensor() if not self.apply_to_layer else wrapped_layer
            output = self.fn(fn_inputs)
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
        new_wrapped = self.input_layers[0].reuse_with(*layers)

        # forward any previous attributes if we're wrapping over other WrapLayer instances
        attr_fwd = self.forward_attributes
        if isinstance(new_wrapped, WrapLayer):
            attr_fwd += new_wrapped.forward_attributes

        if name is None:
            name = self.name

        return WrapLayer(layer=new_wrapped,
                         n_units=self.n_units,
                         fn=self.fn,
                         forward_attributes=attr_fwd,
                         name=name,
                         apply_to_layer=self.apply_to_layer)


class Input(Layer):
    """ Input Layer receives values that can be interpreted as ``Tensor`` or ``SparseTensor``
    """

    def __init__(self, value=None, n_units=None, constant=False, sparse=False, n_active=None, shape=None, dtype=None,
                 name="input"):
        """
        if n_active is not None:
            when connected to a Linear layer, this is interpreted
            as a binary sparse input layer and the linear layer is constructed using the
            Embedding Lookup operator.

            expects: int64 as inputs

        Note:
            if you want to feed a batch of sparse binary features with weights, use SparseInput instead

        Args:
            n_units: number of units in the tensor of this layer
            n_active: number of active units <= n_units
            dtype: type of tensor values
            name: name for the tensor
        """
        if n_active is not None and n_active >= n_units:
            raise ValueError("n_active must be < n_units")

        self.n_active = n_active
        self.shape = shape
        self.constant = constant
        self.sparse = sparse
        # self.updated = True

        if value is not None:
            if n_active is not None:
                self._value = as_tensor(value, dtype=tf.int64)
            else:
                self._value = as_tensor(value, dtype=dtype)

        else:
            self._value = None

        # if self._value.dtype != dtype
        #    self._valuye = tf.cast(self._value,)

        if self._value is not None:
            dtype = self._value.dtype

        super().__init__(None, n_units=n_units, dtype=dtype, name=name)

    def init_state(self):
        layer_state = super().init_state()

        if self._value is not None:
            # if len(self._value.shape) == 0:
            #    self._value = tf.reshape(self._value, [1, 1])
            if len(self._value.shape) > 0:
                self.n_units = self._value.shape[-1]
            else:
                self.n_units = 0

        if self.n_active is not None:
            expected = [None, self.n_active]
        else:
            if self.shape is not None:
                expected = self.shape
                # expected = [None] * len(self.shape[:-1]) + [self.n_units]
            elif self._value is not None:
                if self.n_units > 0:
                    expected = [None] * len(self._value.shape[:-1]) + [self.n_units]
                else:
                    expected = []
            else:
                expected = [None, self.n_units]

        expected = tf.TensorShape(expected)

        if self.shape is not None:
            self.shape = tf.TensorShape(self.shape)
            if not self.shape.is_compatible_with(expected):
                raise ValueError("Invalid shape for Input\n\texpected: {shape}\n\t"
                                 " current: {invalid}".format(shape=expected, invalid=self.shape))
        else:
            self.shape = expected

        if self._value is None and self.shape[-1] is not None:
            if self.dtype is None:
                self.dtype = tf.float32
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
            self._value = [[]]

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
        if self.constant:
            return self._value
        else:
            # self.slot.assign(self._value)
            return self.slot.value()

    @value.setter
    def value(self, x):
        if self.constant:
            raise ValueError("Cannot set the value of a constant Input Layer")
        x = as_tensor(x)
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

    def compute(self, *args):
        with layer_scope(self):
            if self.n_active is not None:
                return txf.sparse_one_hot(self.value, num_cols=self.n_units, dtype=self.dtype)
            else:
                return self.value

    def __str__(self):
        class_name = type(self).__name__
        if self.n_active is not None:
            str_representation = "{layer_name}::{class_name}({n_active}/{n_units},{dtype})[Sparse]".format(
                layer_name=self.scoped_name,
                class_name=class_name,
                n_active=self.n_active,
                n_units=self.n_units,
                dtype=self.dtype)
        else:
            str_representation = "{layer_name}::{class_name}({n_units},{dtype})".format(
                layer_name=self.scoped_name,
                class_name=class_name,
                n_units=self.n_units,
                dtype=self.dtype)

        return str_representation

    def reuse_with(self, *layers, name=None):
        raise AttributeError("Cannot call reuse_with on Input Layer: Input has no input layers")


class Param(Input):
    """ Tensor(value) is an alias for Input(value,constant=True)
        """

    def __init__(self,
                 value,
                 n_units=0,
                 shape=None,
                 dtype=None,
                 name="param"):
        super().__init__(value=value,
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


class Tensor(Input):
    """ Tensor(value) is an alias for Input(value,constant=True)
    """

    def __init__(self,
                 value=None,
                 n_units=None,
                 shape=None,
                 dtype=None,
                 name="tensor"):
        super().__init__(value=value,
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
                 init=None,
                 share_state_with=None,
                 name="variable"):
        self.share_state_with = share_state_with
        self.update_once = update_once
        self.shape = shape
        self.trainable = trainable
        self.resource = resource
        self.init = init if init is not None else tf.initializers.zeros()

        super().__init__(input_layer, n_units, dtype=dtype, name=name)

    def init_state(self):
        layer_state = super().init_state()

        input_layer = self.input_layers[-1] if len(self.input_layers) > 0 else None

        if input_layer is not None:
            if self.n_units is not None and self.n_units != input_layer.n_units:
                raise ValueError("n_units must match input_layer.n_units")
            self.n_units = input_layer.n_units
            self.dtype = input_layer.dtype if self.dtype is None else self.dtype
            self.shape = [1, self.n_units]

            # input_value = input_layer.compute()
            # self.var_shape = input_value.shape.as_list()
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

                layer_state.variable = variable
                layer_state.counter = counter
            else:
                layer_state = self.share_state_with.layer_state

        self.layer_state = layer_state
        return layer_state

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_layer = input_layers[-1] if input_layers else None
        if input_layer is not None:
            input_value = as_layer(input_layer).tensor()

        with layer_scope(self):
            def update():
                self.layer_state.counter.assign_add(1)
                self.layer_state.variable.assign(input_value)
                return self.layer_state.variable.value()

            if input_layer is not None:
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
    def __init__(self, layer, perm=None, n_units=None, name="transpose"):

        self.perm = perm
        self.name = name
        self.n_units = n_units

        super().__init__(input_layers=layer, n_units=self.n_units, dtype=layer.dtype, name=name)

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers
        else:
            input_layers = [as_layer(x) for x in input_layers]

        input_layer = input_layers[0]
        input_value = input_layer.compute()

        with layer_scope(self):
            if not isinstance(input_value, tf.SparseTensor):
                output = tf.transpose(input_value, self.perm)
            else:
                output = tf.sparse.transpose(input_value, self.perm)

            # this is only needed to determine the
            n_units = output.shape.as_list()[-1]

            if self.n_units is not None:
                if self.n_units != n_units:
                    expected = self.n_units
                    raise ValueError("n_units is different from defined n_units:\n"
                                     f"\texpected: {expected}\n"
                                     f"\tn_units: {n_units}")

            return output

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name
        return Transpose(layer=layer, perm=self.perm, name=name)


class Reshape(Layer):
    def __init__(self, layer, shape, name="reshape"):
        """

        Args:
            layer (Layer): an input layer to be reshaped
        """
        self.target_shape = [d if d is not None else -1 for d in shape]
        n_units = self.target_shape[-1] if self.target_shape[-1] > 0 else None
        super().__init__(input_layers=layer, n_units=n_units, dtype=layer.dtype, name=name)

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers
        input_layer = input_layers[0]
        input_value = as_layer(input_layer).compute()

        with layer_scope(self):
            if not isinstance(input_value, tf.SparseTensor):
                output = tf.reshape(input_value, self.target_shape)
            else:
                output = tf.sparse_reshape(input_value, self.target_shape)

        # update n_units
        n_units = output.get_shape().as_list()[-1]
        if self.n_units is None:
            self.n_units = n_units
        elif self.n_units != n_units:
            raise ValueError(
                "n_units changes between computations:\n\tprevious: {}\n\t current: {}".format(self.n_units, n_units))
        return output

    def reuse_with(self, layer, name=None):
        if name is None:
            name = self.name
        return Reshape(layer, self.target_shape, name)


class Linear(Layer):
    """ Linear(input_layer: Layer, n_units, shape=None add_bias=True)


    Fully connected layer that implements a linear transformation of the form :math:`f(x) = Wx + b`

    Args:
            input_layer (Layer): input layer
            n_units (Optional[int]): number of output units
            shape (Optional[Iterable[int]]): shape for the layer weights (not the output). Needed if n_units and
                input_layer.n_units is not known. If add_bias then the bias variable has shape [shape[-1]]
            weight_init: weights (W) initializer function
            shared_weights: variable to be used as linear weights
            shared_bias: variable to be used as a bias
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
                 shape=None,
                 weight_init=tf.initializers.glorot_uniform(),
                 shared_weights=None,
                 shared_bias=None,
                 transpose_weights=False,
                 sparse_weights=False,
                 add_bias=True,
                 bias_init=tf.initializers.zeros(),
                 weight_norm=False,
                 dtype=tf.float32,
                 name="linear",
                 share_state_with=None):

        self.shared_weights = shared_weights
        self.shared_bias = shared_bias
        self.weight_init = weight_init
        self.add_bias = add_bias
        self.share_state_with = share_state_with
        self.transpose_weights = transpose_weights
        self.sparse_weights = sparse_weights
        self.bias_init = bias_init
        self.weight_norm = weight_norm
        self.shape = tuple(as_list(shape)) if shape else None

        if not isinstance(input_layer, Layer):
            input_layer = Input(input_layer, constant=True, dtype=dtype)

        if input_layer.n_units is None or isinstance(input_layer.n_units, tf.Tensor):
            if self.shape is None:
                raise ValueError("Cannot create Linear layer from unknown previous n_units")

        if self.shape is not None:
            if n_units is None:
                n_units = self.shape[-1]
            if self.shape[-1] != n_units:
                raise ValueError("shape[-1] does not match n_units:\n\tshape[-1]: {}"
                                 "\n\tn_units: {}".format(self.shape[-1], n_units))
            if input_layer.n_units is not None:
                if self.shape[0] != input_layer.n_units:
                    raise ValueError("shape[0] does not match input.n_units:\n\tshape[0]: {}"
                                     "\n\tinput.n_units: {}".format(self.shape[0], input_layer.n_units))
        else:
            self.shape = [input_layer.n_units, n_units]

        super().__init__(input_layers=input_layer, n_units=n_units, dtype=dtype, name=name)

    def init_state(self):
        self.layer_state = super().init_state()

        with layer_scope(self):
            input_layer = self.input_layers[0]

            # weights_shape = [input_layer.n_units, self.n_units]

            if self.share_state_with is not None:
                if not isinstance(self.share_state_with, Linear):
                    raise TypeError("Layer can only share variables with other layer of the same type")

                shape = [input_layer.n_units, self.n_units]
                shared_shape = self.share_state_with.weights.get_shape().as_list()
                if self.transpose_weights:
                    shared_shape = shared_shape[::-1]

                if shape != shared_shape:
                    raise ValueError("Can only share variables with layers with the same dimensions: "
                                     "share_state_with is provided but \n"
                                     "self shape: {s0} different from "
                                     "other shape: {s1}".format(s0=shape, s1=shared_shape))

            # if weights are passed, check that their shape matches the layer shape
            if self.shared_weights is not None:
                weight_shape = self.shared_weights.shape

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

            if self.shared_bias is not None:
                bias_shape = self.shared_bias.get_shape().as_list()
                if bias_shape[0] != self.n_units:
                    raise ValueError(
                        "invalid shared bias: number of bias {} does not match number of units {}".format(bias_shape[0],
                                                                                                          self.n_units))

            weights = self.share_state_with.weights if self.share_state_with is not None else self.shared_weights
            if weights is None:
                init_value = self.weight_init(self.shape, dtype=self.dtype)
                weights = tf.Variable(initial_value=init_value,
                                      trainable=True,
                                      dtype=self.dtype,
                                      name="weights")

            self.layer_state.weights = weights

            bias = self.share_state_with.bias if self.share_state_with is not None else self.shared_bias
            if self.add_bias:
                if bias is None:
                    bias = tf.Variable(initial_value=self.bias_init([self.n_units], self.dtype),
                                       name="bias", trainable=True)
            else:
                bias = None
            self.layer_state.bias = bias

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_layer = input_layers[0]
        input_tensor = as_layer(input_layer).compute()
        weights = self.layer_state.weights

        if input_tensor.dtype != weights.dtype:
            raise ValueError("invalid dtype for Linear inputs:\n"
                             "\t expected (weights dtype): {}\n"
                             "\t                 received: {}".format(weights.dtype, input_tensor.dtype))

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
            if self.bias is not None:
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

        Return:
            ``Layer``: a new layer with shared variables with the current layer.

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
                      shared_weights=self.shared_weights,
                      transpose_weights=transpose_weights,
                      sparse_weights=sparse_weights,
                      add_bias=self.add_bias,
                      weight_norm=self.weight_norm,
                      name=name,
                      share_state_with=share_state_with)


class Module(Layer):
    """ Module Layer

    Note:
        The difference between a `Module` and a `Graph` is that a `Module` has a single output just like any other layer
        a Layer graph can be used to group multiple modules. Also the fundamental difference between this and other layers
        is that this is used to group and reuse parts of an existing larger graph.

    Warnings:
        if any path from the layers in inputs does not lead to the output, this is an invalid module
        and an exception is raised when the Module is created

    Args:
        inputs: one or more input layers
        output: output layer
    """

    def __init__(self, inputs, output, name="module"):
        self.inputs = as_list(inputs)
        self.output = output

        try:
            self.graph = Graph.build(self.inputs, self.output)

            # to that we don't need to call reuse on compute with params
            self.graph_compute = self.graph.compile(ord_inputs=self.inputs,
                                                    ord_outputs=self.output)
            if not self.inputs:
                self.inputs = list(self.graph.in_nodes)
        except ValueError as e:
            raise ValueError("Could not build a model with the given endpoints: \n\t{}".format(str(e)))

        super().__init__(input_layers=self.inputs,
                         n_units=output.n_units,
                         dtype=output.dtype,
                         name=name)

    def init_state(self):
        state = super().init_state()

        # add layer to state so that we can retrieve variables of a module etc
        for i, node in enumerate(self.graph.nodes):
            setattr(state, f"layer_{i}", node)

    def compute(self, *input_layers):
        if not input_layers:
            return self.output.compute()
        else:
            input_layers = [as_layer(x).compute() for x in input_layers]
            return self.graph_compute(*input_layers)

    def reuse_with(self, *layers, name=None):
        if name is None:
            name = self.name

        nl = len(layers)
        nm = len(self.input_layers)

        if nl > nm:
            raise ValueError(f"Module has {nm} input layers, {nl} provided")

        layers = [as_layer(x) for x in layers]
        matching_inputs = self.input_layers[:nl]
        mismatch_dtype = list(map(lambda x: x[0].dtype != x[1].dtype, zip(layers, matching_inputs)))
        layers = layers + self.input_layers[nl:]
        if any(mismatch_dtype):
            raise ValueError(f"dtype mismatch in reuse_with:\n"
                             f"\t     expected types: {[x.dtype for x in self.input_layers]}\n"
                             f"\t calling reuse with: {[x.dtype for x in layers]}")

        # maps old inputs to new inputs
        layer_map = dict(zip(self.input_layers, layers))

        dep_iter = self.graph.dependency_iter()
        for node in dep_iter:
            if node not in self.graph.in_nodes:
                new_inputs = [layer_map[x] for x in node.input_layers]
                layer_map[node] = node.reuse_with(*new_inputs)

        new_output = layer_map[self.output]

        # the constructor of Module will trace and compile the new graph

        return Module(inputs=layers, output=new_output, name=name)


class ViewLayer(Layer):
    """ ViewLayer

    Has same shape and inputs as input layer and stores this layer for future reference.
    This means ViewLayer can substitute Layer where layer would be used

    Properties:
        inner_layer (Layer) wrapped by a view
    """

    def compute(self, *args):
        return super().compute(*args)

    def __init__(self, layer, dtype=None, forward_attributes=None, name=None):
        name = "view_{}".format(layer.name) if name is None else name
        dtype = layer.dtype if dtype is None else dtype
        self.inner_layer = layer
        super().__init__(input_layers=layer.input_layers,
                         n_units=layer.n_units,
                         dtype=dtype,
                         name=name)

        self.attr_fwd = as_list(forward_attributes)
        for attr in self.attr_fwd:
            if hasattr(self.inner_layer, attr):
                setattr(self, attr, getattr(self.inner_layer, attr))

    def init_state(self):
        layer_state = super().init_state()
        setattr(layer_state, "inner_layer", self.inner_layer)
        return layer_state


class DropConnect(ViewLayer):
    """ DropConnect

    Args:
            layer (Layer):
            probability (float):
            locked (bool):
            name (str):

    """

    def __init__(self, layer, probability=0.5, locked=True, share_state_with=None, name=None):
        if not isinstance(layer, Linear):
            raise TypeError("DropConnect can only wrap Linear layers: {} found instead".format(layer))

        self.probability = probability
        self.bias = None
        self.weights = None
        self.locked = locked
        self.share_state_with = share_state_with
        name = name if name is not None else "drop_{}".format(layer.name)

        super().__init__(layer, name=name)

    def init_state(self):
        if self.share_state_with is not None:
            self.layer_state = self.share_state_with.layer_state
        else:
            self.layer_state = super().init_state()
            self.layer_state.weight_mask = txf.binary_mask(self.inner_layer.weights, self.probability)
            if self.inner_layer.bias is not None:
                self.layer_state.bias_mask = txf.binary_mask(self.inner_layer.bias)

        return self.layer_state

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_layer = as_layer(input_layers[0])

        with layer_scope(self):
            w = self.inner_layer.weights
            b = self.inner_layer.bias

            drop_w, w_mask = txf.dropout(w, probability=self.probability, random_mask=self.layer_state.weight_mask,
                                         scale=False,
                                         return_mask=True)
            # self.layer_state.weight_mask = w_mask
            drop_b = None
            add_bias = b is not None
            if add_bias:
                drop_b, b_mask = txf.dropout(b, probability=self.probability, random_mask=self.layer_state.bias_mask,
                                             scale=False,
                                             return_mask=True)
                # self.layer_state.bias_mask = b_mask

            new_linear = Linear(input_layer,
                                n_units=self.n_units,
                                shared_weights=drop_w,
                                shared_bias=drop_b,
                                add_bias=add_bias)
            # forward weights and bias
            self.weights = new_linear.weights
            self.bias = new_linear.bias

            return new_linear.compute()

    def reuse_with(self, layer, name=None, locked=None):
        new_layer = self.inner_layer.reuse_with(layer)

        locked = self.locked if locked is None else locked
        name = self.name if name is None else name
        share_state_with = self if locked else None

        return DropConnect(layer=new_layer,
                           probability=self.probability,
                           locked=locked,
                           share_state_with=share_state_with,
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
                 locked=False,
                 seed=None,
                 share_state_with=None,
                 name="dropout"):
        self.seed = seed
        self.probability = probability
        self.scale = scale
        self.noise_shape = noise_shape
        self.locked = locked
        self.share_state_with = share_state_with

        super().__init__(input_layers=input_layer,
                         n_units=input_layer.n_units,
                         dtype=input_layer.dtype,
                         name=name)

    def init_state(self):
        if self.share_state_with is None:
            layer_state = super().init_state()
            layer_state.mask = None
            with layer_scope(self):
                with tf.name_scope(name="random_mask"):
                    if self.noise_shape is not None:
                        keep_prob = 1 - self.probability
                        random_state = tf.random.uniform(self.noise_shape, seed=self.seed,
                                                         dtype=self.input_layers[0].dtype)
                        mask = keep_prob + random_state
                        mask = tf.math.floor(mask, name="binary_mask")

                        layer_state.mask = mask
        else:
            self.layer_state = self.share_state_with.layer_state

        return self.layer_state

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_value = as_layer(input_layers[0]).compute()

        with layer_scope(self):
            if isinstance(input_value, tf.SparseTensor):
                # if input is sparse, noise_shape is not used
                tensor, mask = txf.sparse_dropout(sp_tensor=input_value,
                                                  mask=self.layer_state.mask,
                                                  probability=self.probability,
                                                  scale=self.scale,
                                                  return_mask=True,
                                                  seed=self.seed)

            else:
                tensor, mask = txf.dropout(tensor=input_value,
                                           noise_shape=self.noise_shape,
                                           random_mask=self.layer_state.mask,
                                           probability=self.probability,
                                           scale=self.scale,
                                           return_mask=True,
                                           seed=self.seed)

            if self.layer_state.mask is None:
                self.layer_state.mask = mask

            return tensor

    def reuse_with(self, layer, name=None, locked=None):
        locked = self.locked if locked is None else locked
        name = self.name if name is None else name
        share_state_with = self if locked else None

        return Dropout(layer,
                       probability=self.probability,
                       noise_shape=self.noise_shape,
                       scale=self.scale,
                       locked=locked,
                       seed=self.seed,
                       share_state_with=share_state_with,
                       name=name)


class BaseRNNCell(Layer):
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
        # fills in all previous states\
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

        # needs to be defined on each recurrent cell just as we define self.compute()
        # the default state is the current cell which gives access to its  output tensor
        # self.layer_state =self

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
                 share_state_with: Optional['RNN'] = None,
                 name="rnn_layer"):

        self.cell_proto = cell_proto
        self.share_state_with = share_state_with
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

    def init_state(self):
        layer_state = super().init_state()
        input_seq = self.input_layers[0]

        with layer_scope(self):
            if self.reverse:
                i0 = tf.shape(input_seq)[0] - 1
            else:
                i0 = 0

            x0 = Input(input_seq[i0], constant=True)

            if self.share_state_with is not None:
                cell = self.share_state_with.cell
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

            layer_state.cell = cell
            self.previous_state = cell.previous_state
            self.n_units = cell.n_units

        return layer_state

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers
        input_seq = as_layer(input_layers[0]).compute()
        cell = self.cell

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

            output_ta = output_ta.write(i0, cell.compute())

            def rnn_unroll(t, y, state):
                xt = input_ta.read(t)
                xt = Input(xt, constant=True)
                c = cell.reuse_with(xt, previous_state=state)

                y = y.write(t, c.tensor())
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
                # TODO fix this zero state assign
                # this is no longer necessary because we can assign directly
                # with zero_state.variable.assign(last_state)
                # for zero_state in cell.previous_state:

                updates = [
                    zero_state.reuse_with(last_state, init_from_input=False).tensor()
                    for zero_state, last_state in zip(cell.previous_state, last_state)
                ]

                # TODO unnecessary because assigns are done when we call tensor
                # with tf.control_dependencies(updates):
                out = out.stack()
            else:
                out = out.stack()

            # since the loop outputs a state which is a list
            return [out] + last_state

    def reuse_with(self, input_seq, previous_state=None, regularized=None, reverse=None, stateful=None, name=None):
        name = self.name if name is None else None
        regularized = self.regularized if regularized is None else regularized
        reverse = self.reverse if reverse is None else reverse
        share_state_with = self.share_state_with if self.share_state_with is not None else self
        previous_state = self.previous_state if previous_state is None else previous_state
        stateful = self.stateful if stateful is None else stateful

        return RNN(input_seq=input_seq,
                   cell_proto=self.cell_proto,
                   regularized=regularized,
                   previous_state=previous_state,
                   stateful=stateful,
                   reverse=reverse,
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

            state = Module(inputs=[previous_state, input_layer],
                           output=output,
                           name=self.name + "_h")

            if self.regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            output = Module(inputs=[previous_state, input_layer],
                            output=output,
                            name=self.name + "_output")

            self.output = output
            self.state = [state]

        return layer_state

    def compute(self, input_layer=None, previous_state=None):

        if previous_state and len(previous_state) != len(self.state_size):
            raise ValueError(f"previous state:\n"
                             f"\thas {len(previous_state)} elements\n"
                             f"\texpected {self.state_size}")

        input_layer = self.input_layers[0] if input_layer is None else input_layer
        previous_state = self.previous_state if previous_state is None else tuple(as_list(previous_state))
        output = self.output.compute(previous_state[0], input_layer)

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
            layer: a Layer to be gated
            gate_input: a layer to be used as the gate input
            gate_fn: function for gate
    """

    def __init__(self, layer, gate_input, gate_fn=tf.sigmoid, name="gate"):

        self.gate_fn = gate_fn

        super().__init__(input_layers=[layer, gate_input],
                         n_units=layer.n_units,
                         dtype=tf.float32,
                         name=name)

    def compute(self, input_layer=None, gate_input=None):
        input_layer = input_layer if input_layer is not None else self.input_layers[0]
        gate_input = gate_input if gate_input is not None else self.input_layers[1]

        input_tensor = as_layer(input_layer).compute()
        gate_tensor = as_layer(gate_input).compute()

        with layer_scope(self):
            gate = self.gate_fn(gate_tensor)
            output = txf.apply_gate(input_tensor, gate)

            return output

    def reuse_with(self, input_layer, gate_input=None, name=None):
        if gate_input is None:
            gate_input = self.gate_input

        if name is None:
            name = self.name

        return Gate(layer=input_layer,
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
                 shared_bias=None,
                 shared_weights=None,
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
        self.shared_bias = shared_bias

        n_units = embedding_shape[-1]

        self.batch_size = batch_size

        self.share_state_with = share_state_with
        self.shared_weights = shared_weights

        super().__init__(input_layers=input_layer, n_units=n_units, dtype=dtype, name=name)
        # self.output_shape = [self.shape[0], self.seq_size, self.n_units]

    def init_state(self):
        layer_state = super().init_state()

        # validate shared state
        if self.shared_weights is not None:
            weight_shape = self.shared_weights.get_shape().as_list()
            if self.embedding_shape != weight_shape:
                raise ValueError(
                    "shared weight shape {} and feature shape {} mismatch".format(weight_shape, lookup_shape))

        if self.shared_bias is not None:
            num_bias = self.shared_bias.get_shape().as_list()[-1]
            if self.embedding_shape[0] != num_bias:
                raise ValueError(
                    "number of bias {} and number of feature rows {} mismatch".format(num_bias, lookup_shape[0]))

        if self.share_state_with is not None:
            if not isinstance(self.share_state_with, Lookup):
                raise TypeError("Layer can only share variables with other layer of the same type (Lookup)")

            if self.embedding_shape != self.share_state_with.embedding_shape:
                raise ValueError("Can only share variables with layers with the same feature shape: "
                                 "share_state_with is provided but \n"
                                 "self shape: {s0} different from "
                                 "other shape: {s1}".format(s0=self.embedding_shape,
                                                            s1=self.share_state_with.embedding_shape))

        with layer_scope(self):
            # init weights
            weights = self.share_state_with.weights if self.share_state_with is not None else self.shared_weights
            if weights is None:
                init_value = self.weight_init(self.embedding_shape, dtype=self.dtype)
                weights = tf.Variable(initial_value=init_value,
                                      name="weights",
                                      trainable=True)

            layer_state.weights = weights

            bias = self.share_state_with.bias if self.share_state_with is not None else self.shared_bias
            if self.add_bias:
                if bias is None:
                    bias = tf.Variable(initial_value=self.bias_init([self.embedding_shape[0]], self.dtype),
                                       name="bias", trainable=True)
            else:
                bias = None

            layer_state.bias = bias
        return layer_state

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_layer = input_layers[0]
        input_tensor = as_layer(input_layer).compute()

        # Warn. this validation cannot be done without computing the input
        if isinstance(input_tensor, tf.SparseTensor) and self.seq_size is None:
            raise ValueError("cannot use unknown seq_size with sparse inputs")

        if not isinstance(input_tensor, tf.SparseTensor) and input_layer.dtype not in (tf.int32, tf.int64):
            raise TypeError("invalid input layer dtype {}: should be {} or {}".format(
                input_layer.dtype,
                tf.int32,
                tf.int64
            ))

        if len(input_tensor.shape) > 2:
            raise ValueError("expected 1D/2D input layer")
        elif not isinstance(input_tensor, tf.SparseTensor) and input_layer.n_units is not None:
            if self.seq_size is not None and input_layer.n_units > self.seq_size:
                raise ValueError("input layer n_units ({}) and seq_size ({}) should match for dense input layers \n"
                                 "if n_units < seq_size the lookup will be padded".format(input_layer.n_units,
                                                                                          self.seq_size))

        with layer_scope(self):
            # batch size is unknown for sparse lookups
            # y = xW
            if isinstance(input_tensor, tf.SparseTensor):
                sp_dim = tf.cast(input_tensor.dense_shape[-1], tf.int32)

                # transform.py 1D sparse lookups into 2D sparse lookup with 3 lookups
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
                # tensor = Print(tensor,[tensor[0],tensor[1]],message="padded")

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
                n_units = input_layer.n_units
                if n_units is None:
                    n_units = tf.shape(input_layer.tensor())[-1]

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
                    seq_padding = tf.math.maximum(self.seq_size - input_layer.n_units, 0)
                    padding.append([0, seq_padding])
                else:
                    padding.append([0, 0])

                padding.append([0, 0])
                padding = tf.stack(padding)
                output = tf.pad(output, padding)

        return output

    def as_concat(self):
        def concat_fn(x):
            if self.seq_size is None:
                seq_size = tf.shape(self.input_layers[-1].tensor())[-1]
            else:
                seq_size = self.seq_size

            new_shape = tf.stack([-1, self.n_units * seq_size])
            return tf.reshape(x, new_shape)

        # TODO check if wrap layer can infer n_units
        return WrapLayer(self,
                         n_units=None,
                         fn=concat_fn,
                         forward_attributes=["weights", "bias", "seq_size"],
                         name="concat")

    def permute_batch_time(self):
        return WrapLayer(layer=self,
                         n_units=self.n_units,
                         fn=lambda x: tf.transpose(x, [1, 0, 2]),
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
                      shared_weights=self.shared_weights,
                      weight_init=None,
                      dtype=self.dtype,
                      name=name,
                      share_state_with=share_state_with,
                      batch_padding=self.batch_padding)


# TODO needs tests
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

    def compute(self, layer1=None, layer2=None, gate_input=None):

        layer1 = layer1 if layer1 is not None else self.input_layers[0]
        layer2 = layer2 if layer2 is not None else self.input_layers[1]
        gate_input = gate_input if gate_input is not None else self.input_layers[2]

        input1 = as_layer(layer1).compute()
        input2 = as_layer(layer2).compute()
        gate_input = as_layer(gate_input).compute()

        # TODO check where modules might loose shape when compiled and input to a gate
        #   the bellow verification could not be done

        # if input1.shape[-1] % input2.shape[-1] != 0:
        #     raise ValueError("layers must have the same last dim: {}!={}".format(input1.shape, input2.shape))
        #
        # if input1.shape[-1] % gate_input.shape[-1] != 0:
        #     raise ValueError("the n_units of the input layer {} is not a multiple of gate n_units {}".format(
        #         input1.shape[-1], gate_input.shape[-1]))

        with layer_scope(self):
            gate1 = self.gate_fn(gate_input)
            gate2 = 1 - gate1

            output1 = txf.apply_gate(input1, gate1)
            output2 = txf.apply_gate(input2, gate2)
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
                 u_init=tf.initializers.glorot_uniform(),
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

    def init_state(self):
        layer_state = super().init_state()
        input_layer = self.input_layers[0]
        previous_h = self.previous_state[0]

        with layer_scope(self):
            if self.regularized:
                if self.x_dropout and self.x_dropout > 0:
                    input_layer = self.x_reg(input_layer)
                if self.r_dropout and self.r_dropout > 0:
                    previous_h = self.r_reg(previous_h)

            # TODO
            # if state init is delayed the condition has to be
            # if getattr(layer_state,"w",None) is not None
            # if getattr(layer_state,"u",None) is not None
            # otherwise the else statement would throw an error if w and u where not initialized
            if self.share_state_with is None:
                # reset gate
                # forget / reset bias init to one http://proceedings.mlr.press/v37/jozefowicz15.pdf
                w_r = Linear(input_layer, self.n_units, add_bias=True, bias_init=tf.initializers.ones(), name="w_r")
                u_r = Linear(previous_h, self.n_units, add_bias=False, name="u_r")

                w_c = Linear(input_layer, self.n_units, add_bias=True, weight_init=self.w_init, name="w_c")
                u_c = Linear(previous_h, self.n_units, add_bias=False, weight_init=self.u_init, name="u_c")

                w_z = Linear(input_layer, self.n_units, add_bias=True, name="w_z")
                u_z = Linear(previous_h, self.n_units, add_bias=False, name="u_z")

                w = [w_r, w_c, w_z]
                u = [u_r, u_c, u_z]
            else:
                w = self.share_state_with.layer_state.w
                u = self.share_state_with.layer_state.u

                # in case the layer with wich we share the state has a state wich is regularized
                # TODO store both regularized ViewLayers in the state and normal layers
                # this way we can just retrieve the regularized state
                if not self.regularized:
                    w = list(map(lambda w: w.inner_layer if isinstance(w, ViewLayer) else w, w))
                    u = list(map(lambda u: u.inner_layer if isinstance(u, ViewLayer) else u, u))

                w = [wi.reuse_with(input_layer) for wi in w]
                u = [ui.reuse_with(previous_h) for ui in u]

                w_r, w_c, w_z = w
                u_r, u_c, u_z = u

            if self.regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    w = self.w_reg(*w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    u = self.u_reg(*u)

                w_r, w_c, w_z = w
                u_r, u_c, u_z = u

            # TODO there has to be a more elegant way to update the namespace
            layer_state.w_r = w_r
            layer_state.w_c = w_c
            layer_state.w_z = w_z
            layer_state.u_r = u_r
            layer_state.u_c = u_c
            layer_state.u_z = u_z
            layer_state.w = w
            layer_state.u = u

            r_u_c = Gate(u_c, Add(w_r, u_r, name="add_r"), name="reset_gate")
            candidate = Activation(Add(w_c, r_u_c, name="add_c"), fn=self.activation, name="candidate")
            output = CoupledGate(candidate, previous_h, Add(w_z, u_z, name="add_z"), name="output")

            state = Module(inputs=[previous_h, input_layer],
                           output=output,
                           name=self.name + "_h")
            if self.regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            output = Module(inputs=[previous_h, input_layer],
                            output=output,
                            name=self.name + "_output")

            self.output = output
            self.state = [state]

        return layer_state

    def compute(self, input_layer=None, previous_state=None):
        if previous_state and len(previous_state) != len(self.state_size):
            raise ValueError(f"previous state:\n"
                             f"\thas {len(previous_state)} elements\n"
                             f"\texpected {self.state_size}")

        input_layer = self.input_layers[0] if input_layer is None else input_layer
        previous_state = self.previous_state if previous_state is None else tuple(as_list(previous_state))

        output = self.output.compute(previous_state[0], input_layer)
        return output

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
                 activation=tf.tanh,
                 gate_activation=tf.sigmoid,
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

        self.forget_bias_init = forget_bias_init
        self.gate_activation = gate_activation
        self.output = None
        self.state = None

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
                # forget gate linear
                # http://proceedings.mlr.press/v37/jozefowicz15.pdf bias forget = 1
                w_f = Linear(input_layer, self.n_units, add_bias=True, bias_init=self.forget_bias_init, name="w_f")
                u_f = Linear(previous_h, self.n_units, add_bias=False, name="u_f")

                # input gate linear
                w_i = Linear(input_layer, self.n_units, add_bias=True, name="w_i")
                u_i = Linear(previous_h, self.n_units, add_bias=False, name="u_i")

                # candidate linear
                w_c = Linear(input_layer, self.n_units, add_bias=True, name="w_c")
                u_c = Linear(previous_h, self.n_units, add_bias=False, name="u_c")

                # output gate
                w_o = Linear(input_layer, self.n_units, add_bias=True, name="w_o")
                u_o = Linear(previous_h, self.n_units, add_bias=False, name="u_o")

                w = [w_f, w_i, w_c, w_o]
                u = [u_f, u_i, u_c, u_o]

            else:
                w = self.share_state_with.w
                u = self.share_state_with.u

                # get inner state of dropconnect or other views
                if not self.regularized:
                    w = list(map(lambda w: w.inner_layer if isinstance(w, ViewLayer) else w, w))
                    u = list(map(lambda u: u.inner_layer if isinstance(u, ViewLayer) else u, u))

                w = [wi.reuse_with(input_layer) for wi in w]
                u = [ui.reuse_with(previous_h) for ui in u]

                w_f, w_i, w_c, w_o = w
                u_f, u_i, u_c, u_o = u

            # apply regularizers to weights
            if self.regularized:
                if self.w_dropconnect is not None and self.w_dropconnect > 0:
                    w = self.w_reg(*w)
                if self.u_dropconnect is not None and self.u_dropconnect > 0:
                    u = self.u_reg(*u)

                w_f, w_i, w_c, w_o = w
                u_f, u_i, u_c, u_o = u

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
                memory_state = Module(inputs=[previous_memory,
                                              previous_h,
                                              input_layer],
                                      output=memory_state,
                                      name=self.name + "_memory")

            with tf.name_scope("output"):
                gate_o = Add(w_o, u_o, name="add_o")
                output = Activation(memory_state, fn=self.activation, name="output")
                state = Gate(output, gate_o, gate_fn=self.gate_activation, name="gated_output")

            if self.regularized:
                if self.y_dropout is not None and self.y_dropout > 0:
                    output = self.y_reg(output)

            state = Module(inputs=[input_layer, previous_h, previous_memory],
                           output=state,
                           name=self.name + "_h")

            output = Module(inputs=[input_layer, previous_h, previous_memory],
                            output=output,
                            name=self.name + "_output")

        self.output = output
        self.state = [state, memory_state]
        return layer_state

    def compute(self, input_layer=None, previous_state=None):
        input_layer = self.input_layers[0] if input_layer is None else input_layer
        previous_h, previous_memory = self.previous_state if previous_state is None else previous_state

        output = self.output.compute(input_layer, previous_h, previous_memory)
        return output

    def reuse_with(self, input_layer, previous_state=None, regularized=None, name=None):
        return super().reuse_with(input_layer=input_layer,
                                  previous_state=previous_state,
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
            layer: the input :class:`Layer`
            fn: a function that produces a Tensor and can be called on the tensor produced by the input layer
            name: the layer name
            **kwargs: the keyword arguments for the given function

    """

    def __init__(self, layer, fn=tf.identity, name="activation", **kwargs):

        self.fn = partial(fn, **kwargs)
        self.kw = kwargs
        super().__init__(input_layers=layer, n_units=layer.n_units, dtype=layer.dtype, name=name)

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        input_value = as_layer(input_layers[0]).compute()

        with layer_scope(self):
            if isinstance(input_value, tf.SparseTensor):
                input_value = tf.sparse.to_dense(input_value)
            output = self.fn(input_value)
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
                 n_units=None,
                 dtype=None,
                 weights=None,
                 merge_fn=tf.math.add_n,
                 name="merge"):

        self.weights = weights
        self.merge_fn = merge_fn

        if len(layers) < 1:
            raise Exception("You must provide at least one layer")

        if weights is not None and len(weights) != len(layers):
            raise Exception("len(weights) must be equals to len(layers)")

        super().__init__(input_layers=layers, n_units=n_units, dtype=dtype, name=name)

    def compute(self, *input_layers):
        if not input_layers:
            input_layers = self.input_layers

        outputs = [as_layer(layer).compute() for layer in input_layers]

        if self.weights is not None:
            outputs = [tf.math.scalar_mul(self.weights[i], outputs[i]) for i in
                       range(len(outputs))]

        with layer_scope(self):
            output = self.merge_fn(outputs)

        if self.n_units is None:
            if len(output.shape) > 0:
                self.n_units = output.shape[-1]
        elif self.n_units > 0:
            if output.shape[-1] != self.n_units:
                raise ValueError(
                    f"output n_units {output.shape[-1]} does not match n_units {self.n_units}")

        if self.dtype is None:
            self.dtype = output.dtype
        else:
            if self.dtype != output.dtype:
                output = tf.cast(output, self.dtype)

        return output

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
        layers = list(map(lambda l: as_layer(l), layers))

        n_units = layers[0].n_units
        dtype = layers[0].dtype
        for x in layers[1:]:
            if x.n_units is not None:
                if x.n_units != n_units:
                    raise ValueError("Found layers with different sizes of n_units {}!={} in an Add Layer".format(
                        n_units,
                        x.n_units
                    ))
            if x.dtype is not None:
                if x.dtype != dtype:
                    raise ValueError("Found layers with different dtypes {}!={} in an Add Layer".format(
                        dtype,
                        x.dtype
                    ))

        def merge_add(tensors):
            res = 0
            for tensor in tensors:
                res = res + tensor
            return res

        super().__init__(*layers, n_units=n_units, dtype=dtype, weights=weights, merge_fn=merge_add, name=name)


def as_layer(layer_or_tensor: Union[tf.Tensor, Layer], dtype=None):
    """ Converts a ``Tensor``,``SparseTensor`` or tensor convertible to a ``Layer``

    Args:
        dtype: if not None and different from the input dtype, tries to cast the output layer to the given dtype
        layer_or_tensor: a layer, tensor, or convertible to tensor

    Returns:
        the input ``Layer`` or, a ``Layer`` with the given value
    """
    if isinstance(layer_or_tensor, Layer):
        if dtype is not None and layer_or_tensor.dtype != dtype:
            layer_or_tensor = Lambda(layer_or_tensor, lambda x: tf.cast(x, dtype=dtype), apply_to_layer=False)
        return layer_or_tensor
    else:
        tensor = as_tensor(layer_or_tensor, dtype)
        return Tensor(tensor, dtype=dtype)


# register Layer to tensor conversion
def layer_to_tensor(layer, dtype=None, name=None, as_ref=False):
    name = name if name is not None else layer.name
    with tf.name_scope(name):
        return as_tensor(layer.tensor(), dtype=dtype)


tf.register_tensor_conversion_function(
    base_type=Layer,
    conversion_func=layer_to_tensor,
    priority=100
)


def layer(n_units=None, name="layer", var_list=None):
    """ Decorator for functions that returns a layer prototype

    Returns:
        ``LayerProto`` instance that can be called on layers to create a new layer instance
    """

    def fn_to_proto(fn):
        if isinstance(fn, LayerProto):
            return fn
        return Lambda.proto(fn=fn, n_units=n_units, var_list=var_list, name=name)

    return fn_to_proto


__all__ = [
    "Activation",
    "Lambda",
    "Layer",
    "layer",
    "Linear",
    "Input",
    "Tensor",
    "Param",
    "WrapLayer",
    "VariableLayer",
    "Transpose",
    "Reshape",
    "Add",
    "Module",
    "Gate",
    "CoupledGate",
    "RNNCell",
    "GRUCell",
    "LSTMCell",
    "RNN",
    "Lookup",
    "ToDense",
    "ToSparse"
]
