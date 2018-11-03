"""  Training Module

Contains Module containers to wrap and train neural network models

Contains Learners which are simplified versions of Optimizers. Instead of working with
gradients, they work with delta values which are not necessarily the result of an optimization
process (minimization of a loss function)

This module contains learning procedures different from loss functions used
with gradient descend methods such Winner-Takes-All (WTA) methods for Self-Organising Maps
"""

import os
from abc import ABCMeta, abstractmethod

from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.client.session import Session, InteractiveSession
from tensorflow.python.framework import ops, sparse_tensor
from tensorflow.python.framework.ops import dtypes
from tensorflow.python.ops import array_ops, math_ops, control_flow_ops, logging_ops, clip_ops
from tensorflow.python.ops.gen_state_ops import scatter_sub
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.saver import Saver, import_meta_graph, export_meta_graph
from tensorflow.python.summary import summary
from tensorflow.python import debug as tf_debug

from tensorflow.core.protobuf.config_pb2 import RunOptions, RunMetadata

from tensorx.layers import layers_to_list, Layer, TensorLayer, Input, SparseInput


class VariableUpdater:
    """ Variable Updater.

    Determines how variables are update with dense `Tensor` deltas or `IndexedSlices` deltas.
    """

    def __init__(self, v, use_locking=False):
        self.v = v
        self.use_locking = use_locking

    def update(self, delta):
        if isinstance(delta, ops.Tensor):
            return assign_sub(self.v, delta, self.use_locking)
        else:
            assert isinstance(delta, ops.IndexedSlices), ("Delta ", delta, " is neither a tensor nor IndexedSlices.")

            unique_indices, new_index_positions = array_ops.unique(delta.indices)
            summed_values = math_ops.unsorted_segment_sum(delta.values, new_index_positions,
                                                          array_ops.shape(unique_indices)[0])
            # sum_values = math_ops.cast(sum_values,)
            delta = ops.IndexedSlices(unique_indices, summed_values, delta.dense_shape)

            return scatter_sub(self.v, delta.indices, delta.values, self.use_locking)


class Learner:
    __metaclass__ = ABCMeta

    def __init__(self, var_list, var_updater=VariableUpdater):
        """

        Args:
            var_list: a list of `tf.Variable` to be updated according to the given data
            var_updater:
        """
        self.var_list = var_list
        self.var_updater = var_updater

    def adapt_to(self, data_list, name=None):
        """ Adapts a list of variables to a list of data tensors

        Args:
            data_list: a Tensor or list of tensors from which deltas are computed for the given variables


        Returns:
             An `Operation` that applies the deltas to the variables according to the given data.
        """

        updates = []
        for var, data in zip(self.var_list, data_list):
            deltas_and_vars = self.compute_delta(data)
            vars_with_deltas = [var for var, delta in deltas_and_vars if delta is not None]
            if not vars_with_deltas:
                raise ValueError("No deltas for any variable.")

            updates.append(self.apply_delta(deltas_and_vars))

        return control_flow_ops.group(*updates, name=name)

    @abstractmethod
    def compute_delta(self, data):
        """ Computes the deltas for each variable based on the given data

        Args:
            data: a `Tensor` containing the data used to compute the deltas for the variables

        Returns:
            A list of (delta, variable) pairs. Variable is always present, but
            delta can be `None`.

        """
        return

    def apply_delta(self, deltas_and_vars, name=None):
        """ Apply deltas to variables.

        Args:
            deltas_and_vars: a :obj:`list` of (delta,var)
            name: the name for this op

        Returns:
            An `Operation` that applies the deltas.
        """
        deltas_and_vars = tuple(deltas_and_vars)
        if not deltas_and_vars:
            raise ValueError("No variables provided.")

        converted_deltas_and_vars = []
        for delta, var in deltas_and_vars:
            if delta is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    delta = ops.convert_to_tensor_or_indexed_slices(delta)
                except TypeError:
                    raise TypeError(
                        "Delta must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % delta)
                if not isinstance(delta, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                        "Delta must be a Tensor, IndexedSlices, or None: %s" % delta)

            var_updater = self.var_updater(var)
            converted_deltas_and_vars.append((delta, var, var_updater))

        update_ops = []
        with ops.name_scope(name):
            for delta, var, var_updater in converted_deltas_and_vars:
                if delta is None:
                    continue

                with ops.name_scope("update_" + var.op.name), ops.colocate_with(var):
                    update_ops.append(var_updater.update(delta))

            if len(update_ops) > 1:
                apply_updates = control_flow_ops.group(*update_ops, name=name)
            elif len(update_ops) == 1:
                apply_updates = update_ops[0]
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)

            return apply_updates


""" ********************************************************************************************************************
 Model Container and Model Execution
*********************************************************************************************************************"""


def _default_session():
    """ Returns the default session or a newly created session

    If no default session is available, creates a new session.

    Returns:
        ``Session``: returns the default session if available or a newly created session otherwise.

    """
    session = ops.get_default_session()
    if session is None:
        session = Session()
    return session


def _as_list(elems):
    """ Returns a list from one or multiple elements.

    if one element is passed, returns a list with one element,
    if a list or tuple of elements is passed, returns a list with the elements

    Note: we exclude SparseTensorValue because it is a named tuple
    and we want to feed the whole object as a single data sample if needed

    Args:
        elems: one element, a tuple of elements or a list of elements

    Returns:
        a :obj:`list` with the elements in elems
    """
    if elems is None:
        elems = []
    elif isinstance(elems, (list, tuple)) and not isinstance(elems, sparse_tensor.SparseTensorValue):
        elems = list(elems)
    else:
        elems = [elems]
    return elems


def _get_feedable(inputs):
    feedable = []
    for elem in inputs:
        if hasattr(elem, 'placeholder'):
            feedable.append(elem)
    return feedable


class Param:
    """ Parameter

    The idea with parameters is to provide a way to feed parameters to an optimizer or model.



    """

    def __init__(self, value, dtype=dtypes.float32, name=None):
        self.dtype = dtype
        self.tensor = ops.convert_to_tensor(value)
        self.name = name


class InputParam(Param):
    """ Input Parameter.

    Dev Notes:
        Refactoring Needed?

        TODO think about improving/removing the dependency between InputParam and ModelRunner
        right now it depends on model runner to feed its placeholder with its value, could I make this differently?
        I'm not seeing how right now since the feed_dict is required when calling run. This was created as a way to
        tag parameters as something that might change and needs to be fed (with its own param value). Params are also
        distinct from layers since they are always scalars.

        Dynamic params

        would be params that change according to a given function that might need outside inputs. An example for this
        would be learning rate schedulers. A scheduler could take the current epoch, or the current validation error as
        a basis for the update of its internal value. This way I could isolate the update logic and not pollute the
        running script with this logic. Other than that the input param would work exactly like it has so far.

        dynamic params can have an update function that receives the required parameters as documented and updates the
        internal value. Something simple as that. Calling update just executes the function on the given params.


    Use Case - Optimizer Params:
        An ``InputParam`` can be used to feed a parameter to the optimizer (e.g. learning rate). The optimizer
        would take the param.tensor value and use it as the learning rate. The ``ModelRunner`` will look for
        feedable parameters for the optimizer and try to feed them from the train model method. If none is supplied,
        the param instance is checked for the ``value`` attribute, if this is present, the value is fed to the
        ``InputParam`` placeholder, if not, it throws an error.

        This means one can change the parameters by changing a param.value before calling train on a ``ModelRunner``
        instance. But it will only take effect if the value we're changing belongs to a param that has been configured
        with the ``configure_optimizer`` method in the ``ModelRunner``.
    """

    def __init__(self, dtype=dtypes.float32, init_value=None, name=None):
        self.placeholder = array_ops.placeholder(dtype=dtype, shape=[], name=name)
        super().__init__(self.placeholder, dtype, name)

        self.value = init_value


class DynamicParam(InputParam):
    def __init__(self, init_value=None, dtype=dtypes.float32, update_fn=None, name="param_"):
        super().__init__(dtype=dtype, init_value=init_value, name=name)
        self.update_fn = update_fn

    def update(self, *args, **kwargs):
        if self.update_fn is not None:
            self.value = self.update_fn(*args, **kwargs)


class EvalStepDecayParam(DynamicParam):
    """
    Args:
        init_value: initial value for the dynamic param
        eval_threshold: float value representing the difference between evaluations necessary for the update to occur
        decay_rate: rate through which the param value is reduced `(value = value * decay_rate)`
        decay_threshold: point beyond witch the param value is not reduced `max(value * decay_rate, decay_threshold)`
        less_is_better: if True, evaluation is considered to improve if it decreases, else it is considered to improve
        if it increases

    Attributes:
        eval_history: a list with the evaluation values passed through the update function
        improvement_threshold: float value representing the difference between evaluations necessary for the update to occur
        decay_rate: rate through which the param value is reduced `(value = value * decay_rate)`
        decay_threshold: point beyond witch the param value is not reduced `max(value * decay_rate, decay_threshold)`
    """

    def __init__(self, init_value,
                 improvement_threshold=1.0,
                 less_is_better=True,
                 decay_rate=1.0,
                 decay_threshold=1e-6,
                 dtype=dtypes.float32,
                 name="eval_step_decay_param"):
        self.improvement_threshold = improvement_threshold
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.less_is_better = less_is_better
        self.eval_history = []

        def update_fn(evaluation):
            self.eval_history.append(evaluation)
            value = self.value
            if len(self.eval_history) > 1:
                if self.eval_improvement() <= self.improvement_threshold:
                    value = max(value * self.decay_rate, self.decay_threshold)
            return value

        super().__init__(init_value=init_value, dtype=dtype, update_fn=update_fn, name=name)

    def eval_improvement(self):
        if len(self.eval_history) > 1:
            improvement = self.eval_history[-2] - self.eval_history[-1]
            if not self.less_is_better:
                improvement = -1 * improvement
            return improvement
        else:
            return 0

    def update(self, evaluation):
        """ update.

        Updates the parameter value. It only makes changes to the parameter after then second update.
        The decay decays is only applied if the current evaluation did not improve more than the `eval_threshold`.

        Args:
            evaluation: a float with the current evaluation value
        """
        super().update(evaluation)


class WrapParam(Param):
    """ WrapParam

    Parameter wrapper that augments an existing Param with a function that is always applied to its value.

    Notes:
        This is not in use yet. The original idea was to do something similar to what I have with WrapLayers where
        we can augment a given layer forwarding attributes if necessary. This would be different from a dynamic param
        as it does not contemplate function that might require other external parameters. I guess I should probably
        remove it and replace it with DynamicParam with the update method.

        TODO remove and replace with dynamic param class
    """

    def __init__(self, param, param_fn, dtype=dtypes.float32, name=None):
        if not isinstance(param, Param):
            raise TypeError("expected Param got {} instead".format(type(param)))

        if hasattr(param, "placeholder"):
            self.placeholder = param.placeholder

        if hasattr(param, "value"):
            self.value = param.value

        self.param = param,
        value = param_fn(param.tensor)

        super().__init__(value, dtype, name=name)


class Model:
    """ Model.

    # TODO add support for learners, if the model contains layers that are trainable without using optimisers like SGD

    A `Model` is a container for tensorx graph. It stores the endpoints (input-output) of a model
    and facilitates its visualisation and manipulation.

    Note:
        The basic idea is to show that both loss and eval could be implemented in child Models,
        the default being these to be set to []. It also provides access to inputs, outputs and
        layers which is a list of existing `Layer` instances in the model

    TODO change this documentation accordingly
    Args:
        run_input: a :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        run_output: a :obj:`list` of :class:`Layer` or `Tensor` with the outputs for the model
        train_inputs: a :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        eval_tensors: a single eval tensor or list of tensors

    Attributes:
        loss_tensors: a :obj:`list` of `Tensor` instances with loss functions for the model
        eval_tensors: a :obj:`list` of `Tensor` instances with evaluation functions for the model
        name: a :obj:`str` with the name for this model
        variables: set of all the variables in the model
    """

    def __init__(self,
                 run_in_layers,
                 run_out_layers=None,
                 train_in_layers=None,
                 train_out_layers=None,
                 train_loss_in=None,
                 train_loss_tensors=None,
                 train_loss_weights=None,
                 eval_in_layers=None,
                 eval_out_layers=None,
                 eval_tensors_in=None,
                 eval_tensors=None,
                 name='Model'):
        self.name = name
        # run layers
        self.run_in_layers = _as_list(run_in_layers)
        self.run_out_layers = _as_list(run_out_layers)

        for layer in self.run_in_layers:
            if not isinstance(layer, Layer):
                raise TypeError("run_in_layers expected one or more Layer objects got {} instead".format(type(layer)))

        for layer in self.run_out_layers:
            if not isinstance(layer, Layer):
                raise TypeError("run_out_layers expected one or more Layer objects got {} instead".format(type(layer)))

        # train graph and ops
        if train_in_layers is None:
            self.train_in_layers = self.run_in_layers
        else:
            self.train_in_layers = _as_list(train_in_layers)

        if train_out_layers is None:
            self.train_out_layers = self.run_out_layers
        else:
            self.train_out_layers = _as_list(train_out_layers)

        self.train_loss_in = _as_list(train_loss_in)
        self.train_loss_tensors = _as_list(train_loss_tensors)
        self.train_loss_weights = train_loss_weights

        # eval graph and ops
        if eval_in_layers is None:
            self.eval_in_layers = self.run_in_layers
        else:
            self.eval_in_layers = _as_list(eval_in_layers)

        if eval_out_layers is None:
            self.eval_out_layers = self.run_out_layers
        else:
            self.eval_out_layers = _as_list(eval_out_layers)

        self.eval_tensors_in = _as_list(eval_tensors_in)
        self.eval_tensors = _as_list(eval_tensors)

        # list layers from run, train, and eval
        self.run_layers = layers_to_list(self.run_out_layers)
        self.train_layers = layers_to_list(self.train_out_layers)
        self.eval_layers = layers_to_list(self.eval_out_layers)

        # list graph variables for run, train, and eval
        self.run_vars = {var for layer in self.run_layers for var in layer.variable_names}
        self.train_vars = {var for layer in self.train_layers for var in layer.variable_names}
        self.eval_vars = {var for layer in self.eval_layers for var in layer.variable_names}

    def feedable_run(self):
        """ Returns a list of all inputs in the model that are feedable

        The inputs that use tensorflow placeholders

        Returns:
            a :obj:`list` of input `Layer` instances

        """
        return _get_feedable(self.run_in_layers)

    def feedable_eval(self):
        return _get_feedable(self.eval_in_layers)

    def feedable_eval_tensors(self):
        return _get_feedable(self.eval_tensors_in)

    def feedable_train(self):
        return _get_feedable(self.train_in_layers)

    def feedable_train_tensors(self):
        return _get_feedable(self.train_loss_in)

    def __str__(self):
        lines = ["===== {name}/RUN =====".format(name=self.name)]
        for layer in self.run_layers:
            lines.append(str(layer))
        lines.append("=" * len(lines[0]))

        if len(self.train_layers) > 0:
            lines = ["===== {name}/TRAIN =====".format(name=self.name)]
            for layer in self.train_layers:
                lines.append(str(layer))
            lines.append("=" * len(lines[0]))

        if len(self.eval_layers) > 0:
            lines = ["===== {name}/EVAL =====".format(name=self.name)]
            for layer in self.eval_layers:
                lines.append(str(layer))
            lines.append("=" * len(lines[0]))

        return "\n".join(lines)

    def __repr__(self):
        lines = ["({name}):({inputs})->({outputs})".format(name=self.name,
                                                           inputs=",".join([l.name for l in self.run_in_layers]),
                                                           outputs=",".join([l.name for l in self.run_out_layers]))]

        if self.train_in_layers is not None:
            lines.append("({name}):({inputs})->({outputs})".format(name=self.name,
                                                                   inputs=",".join(
                                                                       [l.name for l in self.train_in_layers]),
                                                                   outputs=",".join(
                                                                       [l.name for l in self.train_out_layers])))

        if self.eval_in_layers is not None:
            lines.append("({name}):({inputs})->({outputs})".format(name=self.name,
                                                                   inputs=",".join(
                                                                       [l.name for l in self.eval_in_layers]),
                                                                   outputs=",".join(
                                                                       [l.name for l in self.eval_out_layers])))

        return "\n".join(lines)

    def has_vars(self):
        return (len(self.run_vars) != 0
                or len(self.train_vars) != 0
                or len(self.eval_vars) != 0)

    def update_state(self):
        """ Updates the model state before evaluation or inference

        If the model train has been called and we call eval or run, this is called before
        any one of those graphs are run.

        Use case:
            Useful to update state variables the inference or evaluation steps might depend upon and we want
            to update only once, and not every time run or eval are called.
        """
        return None


class ModelRunner:
    """ Model Runner

    A model runner takes a model container and facilitates its training and session manager.

    Properties:
        inputs: a single instance or :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        outputs: a single instance or :obj:`list` of :class:`Layer` with the outputs for the model


    """

    def __init__(self, model: Model):
        self.model = model
        self.session = None

        # var inited = ([true|false], session)
        self._var_inited = (None, None)

        # properties for training
        self.optimizer = None
        self.optimizer_params = []
        self.joint_loss = None
        self.var_list = None
        self.train_step = None

        # op for model saving and restoring

        if self.model.has_vars():
            self.saver = Saver()
        self.init_var_op = None

        self.log_writer = None
        self.log_dir = None
        self.runtime_stats = None
        self.run_metadata = None

        self.run_options = None

        self.run_steps = 0
        self.train_steps = 0
        self.eval_steps = 0

        # used to track if train was called since the last time
        # eval or run were called
        # eval or run set this to false
        # train sets this to true
        self.train_called = False

    def set_log_dir(self, log_dir=None):
        self.log_dir = log_dir

        if self.log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), "log")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.log_dir) or not os.path.isdir(self.log_dir):
            raise ValueError("logdir {} does not exist or is not a directory".format(log_dir))

    def set_log_writer(self):
        # log dir changed, change writer
        if self.log_writer is None or not os.path.samefile(self.log_writer.get_logdir(), self.log_dir):
            self.log_writer = FileWriter(self.log_dir, self.session.graph)

    def close_logs(self):
        """ Closes log writers, etc
        """
        self.log_writer.close()

    def _set_vars_inited(self):
        """ Set variables as inited
        Marks the current model as inited
        """
        self._var_inited = (True, self.session)

    def vars_inited(self):
        """ Checks if global variables have been initialised.

        Warning:
            This takes into account the current session under which the model is operating.
            If the session changes,this will return ``False`` since the variables have to be initialised in
            the new session.

        Returns:
            bool: returns true if the variables have been initialised
        """
        inited, init_sess = self._var_inited
        return inited and init_sess == self.session

    def log_graph(self, logdir=None):
        """ outputs the graph meta file to be open in Tensorboard
        Args:
            logdir: path to directory where the graph is to be written

        """
        self.set_session()
        if self.log_dir is not None and logdir is None:
            logdir = self.log_dir
        self.set_log_dir(logdir)
        self.set_log_writer()
        self.log_writer.add_graph(self.session.graph)

    def save_model(self, logdir=None, model_name="model.ckpt", step=None, epoch=None, save_graph=False,
                   write_state=True):
        """ Saves all the variables by default
        # TODO save only some variables this requires init vars to run only
        # on some variables

        Note:
            if no session exists it creates a new default session

        Args:
            write_state: if true writes the checkpoint file with a list of all checkpoints
            save_graph: if true also exports the graph to model_Name.meta
            model_name: name for the model to be saved
            logdir: path to a ckpt file where the model is to be stored
            step: integer or tensor with the current step for the model checkpoint

        """

        if not (self.model.has_vars() or save_graph):
            raise ValueError("The model has no variables to save and save_graph was set to False: Nothing to save")

        if self.session is None:
            self.set_session()

        self.set_log_dir(logdir)
        model_path = os.path.join(self.log_dir, model_name)

        if save_graph:
            meta_path = "{model_path}.meta".format(model_path=model_path)
            export_meta_graph(meta_path)

        if self.model.has_vars():
            self.saver.save(self.session, model_path, step, write_meta_graph=False, write_state=write_state)

    def load_model(self, logdir=None, model_name="model.ckpt", global_step=None, load_graph=False):
        """ Loads the variables on the given path to the current graph, if
        global_step is provided loads that particular checkpoint (if it exists)
        otherwise tries to load the most recent checkpoint with the given name

        Note:
            if a current session does not exist, creates a new session.
            declares the current model as initialised

        Args:
            load_graph:
            global_step: step from which the model should be restored
            logdir: path to the directory where the model is to be saved
            model_name: the path where the model is to be restored
        """
        if self.session is None:
            self.set_session()

        self.set_log_dir(logdir)
        model_path = os.path.join(self.log_dir, model_name)

        if global_step is not None:
            if isinstance(global_step, Variable):
                step = self.session.run(global_step)
            model_path = "{path}-{i}".format(path=model_path, i=step)

        if load_graph:
            meta_path = "{model_path}.meta".format(model_path=model_path)
            self.saver = import_meta_graph(meta_path)

        if self.model.has_vars():
            self.saver.restore(self.session, model_path)
        # we don't need to init vars after loading a model
        self._set_vars_inited()

    def set_session(self, session=None, runtime_stats=False, run_options=None):
        """ Sets the session being used by :class:`Model` class.

        If no session is passed it sets the session as follows:
            1. sets the session to the default session if available
            2. creates a new session and uses it as the default session for the model class.

        Args:
            run_options:
            runtime_stats:
            session: a TensorFlow ``Session``.

        Returns:
            ``Session``: the current ``Session`` being used by the model class.

        """
        if session is not None and not isinstance(session, (Session, InteractiveSession)):
            raise TypeError("Expecting a TensorFlow Session object, got {} instead".format(type(session)))

        if session is None:
            session = _default_session()
        self.session = session

        if self.run_options is None:
            self.run_options = run_options

        if self.runtime_stats is None and runtime_stats:
            self.runtime_stats = runtime_stats
            self.run_metadata = RunMetadata()
            # setup default run options
            if self.run_options is None:
                self.run_options = RunOptions(trace_level=RunOptions.FULL_TRACE)

        return self.session

    def reset_session(self):
        """ Resets the current session.

        Deletes the current session, making the model run under a newly defined session if this is available or creating
        a new session if needed.

        Warning: Note that all the previously initialised variables were initialised under a certain session, this is no
        longer valid for a newly defined session and the whole model runs the variable initialisers again when needed.
        """
        self.session = None

    def close_session(self):
        """ Closes the current tensorflow session.

        If the model is not run inside an externally-defined session, it creates a new session, in which case it should
        be closed.
        """
        self.session.close()

    def init_vars(self):
        """ Initialises all the variables.

        All the variables are initialised in the current session. If no session exists, it tries to find the default
        session. If this is not possible either, it creates a new session which is available in ``self.session``.

        Note:
            In the future perhaps I can initialise only the variables that are defined in the model, for now
            I always end up initialising all the variables anyway. Remember that model is not just a container
            but an utility to reduce the verbose of variable initialisation, session management and training for
            models.
        """
        if self.session is None:
            self.set_session()

        if self.init_var_op is None:
            self.init_var_op = global_variables_initializer()

        self.session.run(self.init_var_op)
        self._var_inited = (True, self.session)

    def run(self, *data, write_summaries=False):
        """ run the model (inference graph)

        Runs the model from the output layers and feeding the data to the input layers

        Uses the current tensorflow ``Session`` to run the model by feeding the given data to the respective inputs.
        the number of data inputs must be the same as the number of inputs.

        Note: it uses the default session if available, if not, creates a new session which is stored in `self.session`

        Args:
            write_summaries: if true adds any summaries created and writes them to the current model folder
            *data: a :obj:`list` or multiple parameters with the data to be fed to each model input

        Returns:
            outputs a :obj:`list` of numpy ``ndarray`` objects

        Raises:
            ValueError: if the number of data items and the number of model inputs are different
        """
        if self.session is None:
            self.set_session()

        if not self.vars_inited() and self.model.has_vars():
            self.init_vars()

        # make sure state is up to date before calling run
        if self.train_called:
            update_op = self.model.update_state()
            if update_op is not None:
                self.session.run(update_op)

        self.train_called = False

        feedable_inputs = self.model.feedable_run()
        n_feedable = len(feedable_inputs)
        n_data = len(data)

        if n_data != n_feedable:
            raise ValueError("data items received {} != {} model feedable inputs".format(n_data, n_feedable))

        feed_dict = {in_layer.placeholder: data for in_layer, data in zip(feedable_inputs, data)}
        output_tensors = [output.tensor for output in self.model.run_out_layers]

        if write_summaries:
            output_tensors = [summary.merge_all()] + output_tensors

        if self.runtime_stats:
            result = self.session.run(output_tensors, feed_dict, options=self.run_options,
                                      run_metadata=self.run_metadata)
            if self.log_dir is None:
                self.set_log_dir()
            self.set_log_writer()
            self.log_writer.add_run_metadata(self.run_metadata, tag="run step {}".format(self.run_steps + 1),
                                             global_step=self.run_steps + 1)
        else:
            result = self.session.run(output_tensors, feed_dict)

        if write_summaries:
            logs, result = result[0], result[1:]
            self.log_writer.add_summary(logs, self.run_steps + 1)

        self.run_steps += 1

        # for convenience if we have a single output layer return the result, not a list of results
        if len(self.model.run_out_layers) == 1:
            result = result[0]
        return result

    def config_optimizer(self, optimizer, params=None, gradient_op=None, global_gradient_op=False, var_list=None):
        """ Configures the model for training

        # TODO add support for other Variable Learners (for SOMs or Free-energy minimisation)
        # TODO add support for gradient monitoring (might be useful to monitor the model)
        # the idea is to add an op that can be applied to the gradients and output in the training method


        Note:
            I suspect we only need to process gradients directly (gradient clipping etc). If the use-case
            arises, we can modify this to accept a function that takes a list of (gradient,variable) tupples
            and returns a list of new  (gradientd,variable) tensors to be applied.

        Gradient OP Example:
            to apply a global gradient op like `tf.clip_by_global_norm`` would require the user to wrap this in a
            function that given a list of gradients produces a list of new gradient tensors:

            gradient_op: [grads] -> [grads]

        Args:
            global_gradient_op: if True applies gradient_op to the entire gradient list,
            if False calls gradient_op for each gradient in the list individually.
            var_list: list o variables modified by the optimizer, if None, the optimizer is applied to
            all variables marked as trainable.
            gradient_op : gradient op is to be applied to each gradient.
            params: a :obj:`list` or single `Param` to be used with the optimizer, the feedable
            parameters should be fed by the same order in the train method

            optimizer: the tensorflow optimiser used to train the model
        """
        self.optimizer = optimizer
        self.optimizer_params = _as_list(params)
        self.var_list = var_list
        loss_weights = self.model.train_loss_weights

        # if more than one loss is passed, create a (optionally weighted) joint loss function
        if len(self.model.train_loss_tensors) > 1 and loss_weights is not None:
            t_losses = ops.convert_to_tensor(self.model.train_loss_tensors)
            loss_weights = math_ops.to_float(loss_weights)
            weighted_losses = math_ops.multiply(t_losses, loss_weights)
            self.joint_loss = math_ops.reduce_sum(weighted_losses)
        else:
            if len(self.model.train_loss_tensors) > 0:
                self.joint_loss = self.model.train_loss_tensors[0]
            else:
                raise ValueError("no loss tensors detected in the given model: \n set train_loss_tensors in Model")

        if gradient_op is not None:
            grads_vars = self.optimizer.compute_gradients(self.joint_loss, var_list=self.var_list)
            gradients, variables = zip(*grads_vars)

            if global_gradient_op:
                new_gradients = gradient_op(gradients)
            else:
                new_gradients = [None if g is None else gradient_op(g) for g in gradients]

            grads_vars = zip(new_gradients, variables)

            self.train_step = self.optimizer.apply_gradients(grads_vars)
        else:
            self.train_step = self.optimizer.minimize(self.joint_loss, var_list=var_list)

    def train(self, data=None, loss_input_data=None, optimizer_params={}, output_loss=False, write_summaries=False):
        """ Trains the model on the given data.

        Uses the configured optimiser and loss functions to train the update the model variables for n
        epochs.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Warning:
            You need to run :func:`config` before calling `train`.

        Args:
            train_step: int with the current train step used to tag runtime metadata
            optimizer_params: values to be fed to the feedable ``Params`` specified in ``config_optimizer``
            data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            loss_input_data: a :obj:`list` of NumPy `ndarray` with the data to be fed to `self.targets`.
        """
        self.train_called = True

        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        if self.train_step is None:
            raise AttributeError("ModelRunner train_step is None, call configure_optimizer before train")

        # =========================
        #   FEED RUN DATA
        # TODO there's a problem with this interface with as_list, if we pass a list as data it already counts
        # as a list, we need to pass an np array for this to work, I should make the check somewhere
        # =========================
        data = _as_list(data)
        feedable_inputs = self.model.feedable_train()
        n_feedable = len(feedable_inputs)
        n_data = len(data)

        if n_data != n_feedable:
            raise ValueError("data items received {}, model requires {} feedable inputs".format(n_data, n_feedable))

        feed_dict = {in_layer.placeholder: data for in_layer, data in zip(feedable_inputs, data)}

        # =======================================
        #   FEED LOSS TENSORS (e.g. with labels)
        # =======================================
        loss_input_data = _as_list(loss_input_data)
        feedable_loss_inputs = self.model.feedable_train_tensors()
        n_feedable_targets = len(feedable_loss_inputs)
        n_targets = len(loss_input_data)

        if n_targets != n_feedable_targets:
            raise ValueError(
                "loss input data received {} != {} model expected loss inputs".format(n_feedable_targets, n_targets))

        target_dict = {loss_input.placeholder: loss_input_data for loss_input, loss_input_data in
                       zip(feedable_loss_inputs, loss_input_data)}

        # =========================
        #   FEED OPTIMIZER PARAMS
        # =========================

        feedable_params = _get_feedable(self.optimizer_params)
        param_dict = {}
        for param in feedable_params:
            if param not in optimizer_params:
                if param.value is not None:
                    param_dict[param.placeholder] = param.value
                else:
                    raise ValueError("expected {p}:value, no value found".format(param.name))
            else:
                param_dict[param.placeholder] = feedable_params[param]

        # MERGE ALL DICTS
        feed_dict.update(target_dict)
        feed_dict.update(param_dict)

        fetches = [self.train_step]
        if output_loss:
            fetches.append(self.model.train_loss_tensors)

        # write logs
        if write_summaries:
            fetches = [summary.merge_all()] + fetches

        # RUNTIME STATISTICS such as compute time, memory etc
        if self.runtime_stats:
            if self.log_dir is None:
                self.set_log_dir()
            self.set_log_writer()

            res = self.session.run(fetches, feed_dict, options=self.run_options, run_metadata=self.run_metadata)

            self.log_writer.add_run_metadata(self.run_metadata,
                                             tag="train step {}".format(self.train_steps + 1),
                                             global_step=self.train_steps + 1)
        else:
            res = self.session.run(fetches, feed_dict)

        if write_summaries:
            logs, result = res[0], res[1:]
            self.log_writer.add_summary(logs, self.train_steps + 1)

        self.train_steps += 1

        if output_loss:
            # res has
            return res[-1]

    def eval(self, data=None, eval_input_data=None, write_summaries=False):
        """ Evaluates the model on the given data.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Args:
            data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            eval_input_data: a :obj:`list` of NumPy `ndarray` with the data to be fed to the evaluation ops.
        """

        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        # make sure state is up to date before calling eval
        if self.train_called:
            update_op = self.model.update_state()
            if update_op is not None:
                self.session.run(update_op)
        self.train_called = False

        data = _as_list(data)

        feedable_inputs = self.model.feedable_eval()
        n_feedable = len(feedable_inputs)
        n_data = len(data)

        if n_data != n_feedable:
            raise ValueError("data items received {} != {} model feedable inputs".format(n_data, n_feedable))

        feed_dict = {in_layer.placeholder: data for in_layer, data in zip(feedable_inputs, data)}

        eval_input_data = _as_list(eval_input_data)
        feedable_eval_inputs = self.model.feedable_eval_tensors()
        n_feedable_targets = len(feedable_eval_inputs)
        n_targets = len(eval_input_data)

        if n_targets != n_feedable_targets:
            raise ValueError(
                "eval input data received {} != {} model expected loss inputs".format(n_feedable_targets, n_targets))

        target_dict = {eval_input.placeholder: loss_input_data for eval_input, loss_input_data in
                       zip(feedable_eval_inputs, eval_input_data)}
        feed_dict.update(target_dict)

        fetches = self.model.eval_tensors

        # write logs
        if write_summaries:
            fetches = [summary.merge_all()] + fetches

        if self.runtime_stats:
            result = self.session.run(fetches, feed_dict, options=self.run_options,
                                      run_metadata=self.run_metadata)
            if self.log_dir is None:
                self.set_log_dir()
            self.set_log_writer()
            self.log_writer.add_run_metadata(self.run_metadata, tag="eval step {}".format(self.eval_step + 1),
                                             global_step=self.eval_step + 1)
        else:
            result = self.session.run(fetches, feed_dict)

        if write_summaries:
            logs, result = result[0], result[1:]
            self.log_writer.add_summary(logs, self.eval_step + 1)

        self.eval_step += 1

        # for convenience if we have a single output layer return the result, not a list of results
        if len(self.model.eval_tensors) == 1:
            result = result[0]
        return result


__all__ = ["Model",
           "ModelRunner",
           "Param",
           "InputParam",
           "DynamicParam",
           "EvalStepDecayParam",
           "WrapParam"]
