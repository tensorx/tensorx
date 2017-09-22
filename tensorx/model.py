""" TensorX model

Provides utilities to put neural network models together. Also facilitates model running, evaluation, and training.

"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.client.session import Session
from tensorx.layers import layers_to_list


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

    def config(self, optimiser, losses, target_inputs=None, loss_weights=1.0):
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


__all__ = ["Model"]
