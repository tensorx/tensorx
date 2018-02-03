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

from tensorflow.python.client.session import Session, InteractiveSession
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import dtypes
from tensorflow.python.ops import array_ops, math_ops, control_flow_ops, logging_ops, clip_ops
from tensorflow.python.ops.gen_state_ops import scatter_sub
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.saver import Saver, import_meta_graph, export_meta_graph

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

    Args:
        elems: one element, a tuple of elements or a list of elements

    Returns:
        a :obj:`list` with the elements in elems
    """
    if elems is None:
        elems = []
    elif isinstance(elems, (list, tuple)):
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
    def __init__(self, value, dtype=dtypes.float32, name=None):
        self.dtype = dtype
        self.tensor = ops.convert_to_tensor(value)
        self.name = name


class InputParam(Param):
    def __init__(self, dtype=dtypes.float32, name=None):
        self.placeholder = array_ops.placeholder(dtype=dtype, shape=[], name=name)
        super().__init__(self.placeholder, dtype, name)


class WrapParam(Param):
    def __init__(self, param, tensor_op, dtype=dtypes.float32, name=None):
        if not isinstance(param, Param):
            raise TypeError("expected Param got {} instead".format(type(param)))

        if hasattr(param, "placeholder"):
            self.placeholder = param.placeholder

        self.param = param,
        value = tensor_op(param.tensor)

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
                 run_out_layers,
                 train_in_layers=None,
                 train_out_layers=None,
                 train_loss_in=None,
                 train_loss_tensors=None,
                 train_loss_weights=1.0,
                 eval_in_layers=None,
                 eval_out_layers=None,
                 eval_tensors_in=None,
                 eval_tensors=None,
                 name='Model'):
        self.name = name
        # run layers
        self.run_in_layers = _as_list(run_in_layers)
        self.run_out_layers = _as_list(run_out_layers)

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


class ModelRunner:
    """ Model Runner

    A model runner takes a model container and facilitates its training and session manager.

    Properties:
        inputs: a single instance or :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        outputs: a single instance or :obj:`list` of :class:`Layer` with the outputs for the model


    """

    def __init__(self, model):
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

    def save_model(self, save_dir=None, model_name="model.ckpt", global_step=None, save_graph=False,
                   write_state=True):
        """ Saves all the variables by default
        # TODO add feature to save only some variables this requires init vars to run only
        # on some variables

        Note:
            if no session exists it creates a new default session

        Args:
            write_state: if true writes the checkpoint file with a list of all checkpoints
            save_graph: if true also exports the graph to model_Name.meta
            model_name: name for the model to be saved
            save_dir: path to a ckpt file where the model is to be stored
            global_step: integer or tensor with the current step for the model checkpoint

        """

        if not (self.model.has_vars() or save_graph):
            raise ValueError("The model has no variables to save and save_graph was set to False: Nothing to save")

        if self.session is None:
            self.set_session()

        if save_dir is None:
            save_dir = os.path.dirname(os.path.realpath(__file__))
        else:
            assert (os.path.exists(save_dir) and os.path.isdir(save_dir))

        model_path = os.path.join(save_dir, model_name)

        if save_graph:
            meta_path = "{model_path}.meta".format(model_path=model_path)
            export_meta_graph(meta_path)

        if self.model.has_vars():
            self.saver.save(self.session, model_path, global_step, write_meta_graph=False, write_state=write_state)

    def load_model(self, save_dir=None, model_name="model.ckpt", global_step=None, load_graph=False):
        """ Loads the variables on the given path to the current graph, if
        global_step is provided loads that particular checkpoint (if it exists)
        otherwise tries to load the most recent checkpoint with the given name

        Note:
            if a current session does not exist, creates a new session.
            declares the current model as initialised

        Args:
            load_graph:
            global_step: step from which the model should be restored
            save_dir: path to the directory where the model is to be saved
            model_name: the path where the model is to be restored
        """
        if self.session is None:
            self.set_session()

        if save_dir is None:
            save_dir = os.path.dirname(os.path.realpath(__file__))
        else:
            assert (os.path.exists(save_dir) and os.path.isdir(save_dir))

        model_path = os.path.join(save_dir, model_name)

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

    def set_session(self, session=None):
        """ Sets the session being used by :class:`Model` class.

        If no session is passed it sets the session as follows:
            1. sets the session to the default session if available
            2. creates a new session and uses it as the default session for the model class.

        Args:
            session: a tensorflow ``Session``.

        Returns:
            ``Session``: the current ``Session`` being used by the model class.

        """
        if session is not None and not isinstance(session, (Session, InteractiveSession)):
            raise TypeError("Expecting a tensorflow Session object, got {} instead".format(type(session)))

        if session is None:
            session = _default_session()
        self.session = session
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

    def run(self, *data):
        """ run the model (inference graph)

        Runs the model from the output layers and feeding the data to the input layers

        Uses the current tensorflow ``Session`` to run the model by feeding the given data to the respective inputs.
        the number of data inputs must be the same as the number of inputs.

        Note: it uses the default session if available, if not, creates a new session which is stored in `self.session`

        Args:
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

        feedable_inputs = self.model.feedable_run()
        n_feedable = len(feedable_inputs)
        n_data = len(data)

        if n_data != n_feedable:
            raise ValueError("data items received {} != {} model feedable inputs".format(n_data, n_feedable))

        feed_dict = {in_layer.placeholder: data for in_layer, data in zip(feedable_inputs, data)}
        output_tensors = [output.tensor for output in self.model.run_out_layers]
        result = self.session.run(output_tensors, feed_dict)

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
        if len(self.model.train_loss_tensors) > 1:
            t_losses = ops.convert_to_tensor(self.model.train_loss_tensors)
            loss_weights = math_ops.to_float(loss_weights)
            weighted_losses = math_ops.multiply(t_losses, loss_weights)
            self.joint_loss = math_ops.reduce_sum(weighted_losses)
        else:
            self.joint_loss = self.model.train_loss_tensors[0]

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

    def train(self, data=None, loss_input_data=None, optimizer_param_values=None):
        """ Trains the model on the given data.

        Uses the configured optimiser and loss functions to train the update the model variables for n
        epochs.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Warning:
            You need to run :func:`config` before calling `train`.

        Args:
            optimizer_param_values: values to be fed to the feedable ``Params`` specified in ``config_optimizer``
            data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            loss_input_data: a :obj:`list` of NumPy `ndarray` with the data to be fed to `self.targets`.
        """
        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        # =========================
        #   FEED DATA
        # =========================
        data = _as_list(data)
        feedable_inputs = self.model.feedable_train()
        n_feedable = len(feedable_inputs)
        n_data = len(data)

        if n_data != n_feedable:
            raise ValueError("data items received {} != {} model feedable inputs".format(n_data, n_feedable))

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
        param_values = _as_list(optimizer_param_values)
        feedable_params = _get_feedable(self.optimizer_params)
        if len(param_values) != len(feedable_params):
            raise ValueError(
                "received {n_params} optimizer parameter values, expected {n_feedable}".format(
                    n_params=len(param_values),
                    n_feedable=len(feedable_params))
            )

        param_dict = {param.placeholder: value for param, value in zip(feedable_params, param_values)}

        # MERGE ALL DICTS

        feed_dict.update(target_dict)
        feed_dict.update(param_dict)

        self.session.run(self.train_step, feed_dict)

    def eval(self, data=None, eval_input_data=None):
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

        result = self.session.run(self.model.eval_tensors, feed_dict)

        # for convenience if we have a single output layer return the result, not a list of results
        if len(self.model.eval_tensors) == 1:
            result = result[0]
        return result


__all__ = ["Model",
           "ModelRunner",
           "Param",
           "InputParam",
           "WrapParam"]
