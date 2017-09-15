""" TensorX model

Provides utilities to put neural network models together. Also facilitates model running, evaluation, and training.

"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


class Model:
    """ Model
            A model is a container for an acyclic network graph. It stores the endpoints (input-output) of a neural
            network and facilitates model training and evaluation.

            Args:
             inputs: a :obj:`list` of :class:`Input` or :class:`SparseInput` with the inputs for the model

             outputs: a :obj:`list` of :class:`Layer` with the outputs for the model
    """

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        # properties for training
        self.optimiser = None
        self.losses = None
        self.targets = None
        self.loss_weights = 1.0
        self.joint_loss = None

    def run(self, *data, session=None):
        """ run the model

        Uses a tensorflow session to run the model by feeding the given data to the respective model inputs.
        the number of data inputs must be the same as the number of inputs.

        Args:
            *data: a :obj:`list` or multiple parameters with the data to be fed to each model input
            session: an existing tensorflow session object. Uses the default session of nothing is provided

        Returns:
            outputs a :obj:`list` of numpy ``ndarray`` objects

        Raises:
            ValueError: if the number of data items and the number of model inputs are different
        """
        n_inputs = len(self.inputs)
        n_data = len(data)

        if n_data != n_inputs:
            raise ValueError("data items received {} != {} model inputs".format(n_data, n_inputs))
        if session is None:
            session = ops.get_default_session()

        feed_dict = {in_layer.tensor: data for in_layer, data in zip(self.inputs, data)}
        output_tensors = [output.tensor for output in self.outputs]

        return session.run(output_tensors, feed_dict)

    def config(self, optimiser, losses, targets=None, loss_weights=1.0):
        """ Configures the model for training

        Args:
            losses: a :obj:`list` of loss `Tensor` instances to be used to train the model variables
            targets: a :obj:`list` of input layers that will be used with the loss function
            optimiser: the tensorflow optimiser used to train the model
            loss_weights: weights used to create a join loss if we configure the model with multiple losses

        """

        self.optimiser = optimiser
        self.losses = losses
        self.targets = targets
        self.loss_weights = loss_weights

        # the default behaviour is to create a (optionally weighted) joint loss function
        t_losses = ops.convert_to_tensor(losses)
        loss_weights = math_ops.to_float(loss_weights)
        weighted_losses = math_ops.multiply(t_losses, loss_weights)
        self.joint_loss = math_ops.reduce_sum(weighted_losses)

    def train(self, data, targets=None, n_epochs=1, session=None):
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

        Returns:

        """
        if session is None:
            session = ops.get_default_session()

        train_step = self.optimiser.minimize(self.joint_loss)

        feed_dict = {in_layer.tensor: data for in_layer, data in zip(self.inputs, data)}
        if targets is not None:
            label_dict = {target.tensor: label for target, label in zip(self.targets, targets)}
            feed_dict = {**feed_dict, **label_dict}

        for epoch in range(n_epochs):
            session.run(train_step, feed_dict)
