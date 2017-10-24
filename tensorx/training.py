"""  Training Module

Contains Module containers to wrap and train neural network models

Contains Learners which are simplified versions of Optimizers. Instead of working with
gradients, they work with delta values which are not necessarily the result of an optimization
process (minimization of a loss function)

This module contains learning procedures different from loss functions used
with gradient descend methods such Winner-Takes-All (WTA) methods for Self-Organising Maps
"""

from abc import ABCMeta, abstractmethod

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, control_flow_ops
from tensorflow.python.ops.gen_state_ops import scatter_sub
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.client.session import Session

from tensorx.layers import layers_to_list


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

    def __init__(self, var_updater=VariableUpdater):
        self.var_updater = var_updater

    def adapt_to(self, var_list, data_list, combined=True, name=None):
        """ Adapts a list of variables to a list of data instances

        Args:
            data_list: a Tensor or list of tensors from which deltas are computed for the given variables
            var_list: a list of `tf.Variable` to be updated according to the given data

        Returns:
             An `Operation` that applies the deltas to the variables according to the given data.
        """

        updates = []
        for var, data in var_list, data_list:
            deltas_and_vars = self.compute_delta(data, var)
            vars_with_deltas = [var for var, delta in deltas_and_vars if delta is not None]
            if not vars_with_deltas:
                raise ValueError("No deltas for any variable.")

            updates.append(self.apply_delta(deltas_and_vars))

        return control_flow_ops.group(*updates, name=name)

    @abstractmethod
    def compute_delta(self, data, var_list):
        """ Computes the deltas for each variable based on the given data

        Args:
            data: a `Tensor` containing the data used to compute the deltas for the variables
            var_list: list or tuple of `tf.Variable` to update

        Returns:
            A list of (delta, variable) pairs. Variable is always present, but
            delta can be `None`.

        """
        raise NotImplemented

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
 Models
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
    if isinstance(elems, (list, tuple)):
        elems = list(elems)
    else:
        elems = [elems]
    return elems


class Model:
    """ Model

    A model is a container for an network graph. It stores the endpoints (input-output) of a model
    and facilitates its training and evaluation.

    Properties:
        inputs: a single instance or :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        outputs: a single instance or :obj:`list` of :class:`Layer` with the outputs for the model

    Args:
        inputs: a :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model
        outputs: a :obj:`list` of :class:`Layer` with the outputs for the model
    """

    def __init__(self, inputs, outputs):

        self.inputs = _as_list(inputs)
        self.outputs = _as_list(outputs)

        self.session = None

        # var inited = ([true|false], session)
        self._var_inited = (None, None)

        # properties for training
        self.optimiser = None
        self.losses = None
        self.targets = None
        self.loss_weights = 1.0
        self.joint_loss = None
        self.train_step = None

        self.layers = layers_to_list(self.outputs)

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
        if session is not None and not isinstance(session, Session):
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

        self.session.run(global_variables_initializer())
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

    def run(self, *data):
        """ run the model

        Uses a tensorflow ``Session`` to run the model by feeding the given data to the respective model inputs.
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

        if not self.vars_inited():
            self.init_vars()

        n_inputs = len(self.inputs)
        n_data = len(data)

        if n_data != n_inputs:
            raise ValueError("data items received {} != {} model inputs".format(n_data, n_inputs))

        feed_dict = {in_layer.tensor: data for in_layer, data in zip(self.inputs, data)}
        output_tensors = [output.tensor for output in self.outputs]
        result = self.session.run(output_tensors, feed_dict)
        if len(self.outputs) == 1:
            result = result[0]
        return result

    def config_optimisers(self, optimiser, losses, target_inputs=None, loss_weights=1.0):
        """ Configures the model for training

        Args:
            losses: a :obj:`list` or single loss `Tensor` instances to be used to train the model variables
            target_inputs: a :obj:`list` or single input layers that will be used with the loss function
            optimiser: the tensorflow optimiser used to train the model
            loss_weights: weights used to create a join loss if we configure the model with multiple losses

        """
        self.losses = _as_list(losses)
        self.target_inputs = _as_list(target_inputs)

        self.optimiser = optimiser
        self.loss_weights = loss_weights

        # the default behaviour is to create a (optionally weighted) joint loss functionz
        if len(self.losses) > 1:
            t_losses = ops.convert_to_tensor(self.losses)
            loss_weights = math_ops.to_float(loss_weights)
            weighted_losses = math_ops.multiply(t_losses, loss_weights)
            self.joint_loss = math_ops.reduce_sum(weighted_losses)
        else:
            self.joint_loss = self.losses[0]

        self.train_step = self.optimiser.minimize(self.joint_loss)

    def config_learners(self, learner_data):
        """

        Args:
            learner_data: a :obj:`list` of `(Learner, data)` pairs to be used with the model

        Returns:

        """

    def train(self, data, targets=None, n_epochs=1):
        """ Trains the model on the given data.

        Uses the configured optimiser and loss functions to train the update the model variables for n
        epochs.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Warning:
            You need to run :func:`config` before calling `train`.

        Args:
            data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            targets: a :obj:`list` of NumPy `ndarray` with the data to be fed to `self.targets`.
            n_epochs: number of times the training op is run on the model
        """
        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        data = _as_list(data)

        n_data = len(data)
        n_inputs = len(self.inputs)

        if n_data != n_inputs:
            raise ValueError("data items received {} != {} model inputs".format(n_data, n_inputs))

        feed_dict = {in_layer.tensor: data for in_layer, data in zip(self.inputs, data)}
        if targets is not None:
            targets = _as_list(targets)
            n_targets = len(targets)
            n_target_inputs = len(self.target_inputs)
            if n_targets != n_target_inputs:
                raise ValueError(
                    "target items received {} != {} model target inputs".format(n_targets, n_target_inputs))

            label_dict = {target.tensor: label for target, label in zip(self.target_inputs, targets)}
            feed_dict.update(label_dict)

        for epoch in range(n_epochs):
            self.session.run(self.train_step, feed_dict)


__all__ = ["Model", "Learner"]
