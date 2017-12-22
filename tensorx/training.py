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
from typing import Tuple, Union, List

from tensorflow.python.client.session import Session, InteractiveSession
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, control_flow_ops
from tensorflow.python.ops.gen_state_ops import scatter_sub
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.saver import Saver, import_meta_graph, export_meta_graph

from tensorx.layers import layers_to_list, Layer, TensorLayer


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


class Model:
    """ Model.

    # TODO add support for learners, if the model contains layers that are trainable without using optimisers like SGD

    A `Model` is a container for tensorx graph. It stores the endpoints (input-output) of a model
    and facilitates its visualisation and manipulation.

    Note:
        The basic idea is to show that both loss and eval could be implemented in child Models,
        the default being these to be set to []. It also provides access to inputs, outputs and
        layers which is a list of existing `Layer` instances in the model

    Args:
        inputs: a :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        outputs: a :obj:`list` of :class:`Layer` with the outputs for the model
        loss_tensors: a single loss tensor or list of loss tensors
        eval_tensors: a single eval tensor or list of tensors

    Attributes:
        loss_tensors: a :obj:`list` of `Tensor` instances with loss functions for the model
        eval_tensors: a :obj:`list` of `Tensor` instances with evaluation functions for the model
        name: a :obj:`str` with the name for this model
        variables: set of all the variables in the model
    """

    def __init__(self, inputs, outputs, loss_tensors=None, eval_tensors=None, name='Model'):
        self.name = name
        self.inputs = _as_list(inputs)
        self.outputs = _as_list(outputs)
        self.layers = layers_to_list(outputs)

        self.variables = {var for layer in self.layers for var in layer.variable_names}

        self.loss_tensors = _as_list(loss_tensors)
        self.eval_tensors = _as_list(eval_tensors)

    def feedable_inputs(self):
        """ Returns a list of all inputs in the model that are feedable

        The inputs that use tensorflow placeholders

        Returns:
            a :obj:`list` of input `Layer` instances

        """
        return [input_layer for input_layer in self.inputs if not isinstance(input_layer, TensorLayer)]

    def __str__(self):
        lines = ["===== {name} =====".format(name=self.name)]
        for layer in self.layers:
            lines.append(str(layer))

        lines.append("=" * len(lines[0]))
        return "\n".join(lines)

    def __repr__(self):
        return "({name}):({inputs})->({outputs})".format(name=self.name,
                                                         inputs=",".join([l.name for l in self.inputs]),
                                                         outputs=",".join([l.name for l in self.outputs]))


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
        self.optimiser = None
        self.losses = model.loss_tensors
        self.targets = None
        self.loss_weights = 1.0
        self.joint_loss = None
        self.train_step = None
        self.target_labels = None

        # op for model saving and restoring

        if len(model.variables) != 0:
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
        if len(self.model.variables) == 0 and not save_graph:
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

        if len(self.model.variables) > 0:
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

        if len(self.model.variables) > 0:
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
        """ run the model

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

        if not self.vars_inited() and len(self.model.variables) > 0:
            self.init_vars()

        feedable_inputs = self.model.feedable_inputs()
        n_feedable = len(feedable_inputs)
        n_data = len(data)

        if n_data != n_feedable:
            raise ValueError("data items received {} != {} model feedable inputs".format(n_data, n_feedable))

        feed_dict = {in_layer.tensor: data for in_layer, data in zip(feedable_inputs, data)}
        output_tensors = [output.tensor for output in self.model.outputs]
        result = self.session.run(output_tensors, feed_dict)

        # for convenience if we have a single output layer return the result, not a list of results
        if len(self.model.outputs) == 1:
            result = result[0]
        return result

    def config_training(self, optimiser, losses=None, target_labels=None, loss_weights=1.0):
        """ Configures the model for training

        # TODO add support for other Variable Learners (for SOMs or Free-energy minimisation)

        Args:
            losses: a :obj:`list` or single loss `Tensor` instances to be used to train the model variables
            target_labels: a :obj:`list` or single input layers that will be used with the loss function
            optimiser: the tensorflow optimiser used to train the model
            loss_weights: weights used to create a join loss if we configure the model with multiple losses
            learner: in case the model uses a learner function to change variables, include it here

        """
        # if losses are provided override the model losses
        if losses is not None:
            self.losses = _as_list(losses)
        self.target_labels = _as_list(target_labels)

        self.optimiser = optimiser
        self.loss_weights = loss_weights

        # if more than one loss is passed, create a (optionally weighted) joint loss function
        if len(self.losses) > 1:
            t_losses = ops.convert_to_tensor(self.losses)
            loss_weights = math_ops.to_float(loss_weights)
            weighted_losses = math_ops.multiply(t_losses, loss_weights)
            self.joint_loss = math_ops.reduce_sum(weighted_losses)
        else:
            self.joint_loss = self.losses[0]

        self.train_step = self.optimiser.minimize(self.joint_loss)

    def train(self, data=None, targets=None):
        """ Trains the model on the given data.

        Uses the configured optimiser and loss functions to train the update the model variables for n
        epochs.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Warning:
            You need to run :func:`config` before calling `train`.

        Args:
            data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            targets: a :obj:`list` of NumPy `ndarray` with the data to be fed to `self.targets`.
        """
        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        data = _as_list(data)
        if len(data) > 0:
            feedable_inputs = self.model.feedable_inputs()
            n_feedable = len(feedable_inputs)
            n_data = len(data)

            if n_data != n_feedable:
                raise ValueError("data items received {} != {} model feedable inputs".format(n_data, n_feedable))

            feed_dict = {in_layer.tensor: data for in_layer, data in zip(feedable_inputs, data)}

            if targets is not None:
                targets = _as_list(targets)
                n_targets = len(targets)
                target_labels = len(self.target_labels)
                if n_targets != target_labels:
                    raise ValueError(
                        "target items received {} != {} model target inputs".format(n_targets, target_labels))

                label_dict = {target.tensor: label for target, label in zip(self.target_labels, targets)}
                feed_dict.update(label_dict)

        self.session.run(self.train_step, feed_dict)


__all__ = ["Model", "ModelRunner"]
