""" TensorX model

Provides utilities to put neural network models together. Also facilitates model running, evaluation, and training.

"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.client.session import Session


def _default_session():
    session = ops.get_default_session()
    if session is None:
        session = Session()
    return session


def _as_list(elems):
    """ returns a list from the given element(s)

    Args:
        elems: one or more objects

    Returns:
        a list with the elements in elems
    """
    if isinstance(elems, (list, tuple)):
        elems = list(elems)
    else:
        elems = [elems]
    return elems


class Model:
    """ Model

    A model is a container for an acyclic network graph. It stores the endpoints (input-output) of a neural
    network and facilitates model training and evaluation.

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

    def set_session(self, session=None):
        if session is None:
            session = _default_session()
        self.session = session
        return self.session

    def reset_session(self):
        self.session = None

    def init_vars(self):
        """ Initialises all the variables
        if session was not set or default
        """
        if self.session is None:
            self.set_session()

        self.session.run(global_variables_initializer())
        self._var_inited = (True, self.session)

    def vars_inited(self):
        inited, init_sess = self._var_inited
        return inited and init_sess == self.session

    def run(self, *data):
        """ run the model

        Uses a tensorflow ``Session`` to run the model by feeding the given data to the respective model inputs.
        the number of data inputs must be the same as the number of inputs.

        Args:
            *data: a :obj:`list` or multiple parameters with the data to be fed to each model input
            session: an existing tensorflow session object. Uses the default session of nothing is provided

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

    def config(self, optimiser, losses, targets=None, loss_weights=1.0):
        """ Configures the model for training

        Args:
            losses: a :obj:`list` or single loss `Tensor` instances to be used to train the model variables
            targets: a :obj:`list` or single input layers that will be used with the loss function
            optimiser: the tensorflow optimiser used to train the model
            loss_weights: weights used to create a join loss if we configure the model with multiple losses

        """
        self.losses = _as_list(losses)
        self.targets = _as_list(targets)

        self.optimiser = optimiser
        self.loss_weights = loss_weights

        # the default behaviour is to create a (optionally weighted) joint loss function
        t_losses = ops.convert_to_tensor(losses)
        loss_weights = math_ops.to_float(loss_weights)
        weighted_losses = math_ops.multiply(t_losses, loss_weights)
        self.joint_loss = math_ops.reduce_sum(weighted_losses)

    def train(self, data, targets=None, n_epochs=1):
        """ Trains the model on the given data.

        Uses the configured optimiser and loss functions to train the update the model variables for n
        epochs.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.



        Warning:
            You need to run :method:`config` before calling `train`.

        Args:
            data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            targets: a :obj:`list` of NumPy `ndarray` with the data to be fed to `self.targets`.
            n_epochs: number of times the training op is run on the model
            session: a tensorflow session

        """
        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        train_step = self.optimiser.minimize(self.joint_loss)

        feed_dict = {in_layer.tensor: data for in_layer, data in zip(self.inputs, data)}
        if targets is not None:
            label_dict = {target.tensor: label for target, label in zip(self.targets, targets)}
            feed_dict = {**feed_dict, **label_dict}

        for epoch in range(n_epochs):
            self.session.run(train_step, feed_dict)
