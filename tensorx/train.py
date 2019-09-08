import tensorflow as tf
import tensorx as tx
from tensorx.utils import as_list, Graph
import itertools
import inspect


class Model:
    """
    Args:
        train_loss: either a callable, layer, or dictionary mapping a callable or layer to the target outputs
        train_inputs: defaults to run inputs, if loss is provided you can either supply the inputs to the train graph
            that include the loss, or let the Model create inputs for you.
    """

    def __init__(self,
                 run_outputs,
                 run_inputs=None,
                 train_outputs=None,
                 train_inputs=None,
                 train_loss=None,
                 eval_inputs=None,
                 eval_outputs=None,
                 name='Model',
                 ):
        self.name = name

        self.run_inputs = as_list(run_inputs)
        self.run_outputs = as_list(run_outputs)
        self.train_inputs = as_list(train_inputs)
        self.train_outputs = as_list(train_outputs)
        self.eval_inputs = as_list(eval_inputs)
        self.eval_outputs = as_list(eval_outputs)

        self.run_graph: Graph = Graph.build(run_inputs, run_outputs)

        if not isinstance(train_loss, tx.Layer):
            raise TypeError(f"Invalid train_loss type\n"
                            f"\t expected: Layer"
                            f"\t actual: {type(train_loss)}")

        self.train_loss = train_loss

        self.train_graph: Graph = Graph.build(self.train_inputs, self.train_outputs)
        self.eval_graph: Graph = Graph.build(self.eval_inputs, self.eval_outputs)

        self.run_fn = self.run_graph.compile()
        # TODO does this include the loss function?
        self.train_fn = self.train_graph.compile()

        self.optimizers = []

    @property
    def trainable_variables(self):
        return list(itertools.chain(*(layer.trainable_vars for layer in self.train_graph.nodes)))

    def _create_train_op(self):
        @tf.function
        def train_step(*args):
            with tf.GradientTape() as tape:
                # predictions = self.train_fn(args)
                # TODO modify the train graph with loss output node only
                loss = self.loss(*args)  # (labels, predictions)
            grads = tape.gradient(loss, self.trainable_variables
            # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # grads = tf.gradients(loss, params)
            # for grad, param in zip(grads, params):

            # if "clipglobalnorm" in config:
            #     grads = [tf.clip_by_global_norm(g, config["clipglobalnorm"]) for g in grads]
            #
            # if "clipnorm"):
            #     grads = [tf.clip_by_norm(g, self.clipnorm) for g in grads]
            # if hasattr(self, "clipvalue"):
            #     grads = [
            #         clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
            #         for g in grads
            #
            #      ]

    def add_optimizer(self, optimizer, **config):
        """
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Optimizer

        Optimizer Hyper parameters:
            arguments passed to the optimizer constructor. They can be either regular Python values,
            tensors, or callables. If they are callable, the callable will be called during
            apply_gradients() to get the value for the hyper parameter.
        Args:
            optimizer_cls: optimizer class
            **opt_kwargs:

        Returns:

        """
        if isinstance(optimizer, tf.optimizers.Optimizer):
            # attributes can be set on optimizers
            # https://github.com/tensorflow/tensorflow/blob/25006be096cd4f0242a3b979fb212bbd192127b3/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L553
            opt = optimizer
            for attr in config:
                setattr(opt, attr, config[attr])
        elif issubclass(optimizer, tf.optimizers.Optimizer):
            opt = optimizer.from_config(config)
        else:
            raise TypeError(f"optimizer_cls should be an instance of subclass of tf.optimizers.Optimizer:\n"
                            f"\tgot {type(optimizer)} instead.")

    def run(self, data, outputs=None):
        """

        Args:
            data: a dict with {``Layer``:value}
            outputs: [Optional] target outputs to be evaluated (from the run_graph)

        Returns:
            list of value with results of computation for each output in the model

        """
        if data is not None:
            if not isinstance(data, dict):
                inputs = [x for x in self.run_graph.in_nodes if isinstance(x, tx.Input) and not x.constant]
                data_feed = dict(zip(inputs, data))

            data_feed = {key: data_feed[key]
                         for key in data_feed.keys()
                         if isinstance(key, tx.Layer)}

            # TODO automatic update ops if train has been called
