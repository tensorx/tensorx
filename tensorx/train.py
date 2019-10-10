import tensorflow as tf
import tensorx as tx
from tensorx.utils import as_list, Graph
import itertools
import inspect
import logging

logger = logging.getLogger('tensorx')


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
        self.compiled_run = None

        if not isinstance(train_loss, tx.Layer):
            raise TypeError(f"Invalid train_loss type\n"
                            f"\t expected: Layer"
                            f"\t actual: {type(train_loss)}")

        self.train_loss = train_loss

        self.train_graph: Graph = Graph.build(self.train_inputs, self.train_outputs + [self.train_loss])
        self.eval_graph: Graph = Graph.build(self.eval_inputs, self.eval_outputs)

        self.run_fn = self.run_graph.compile()
        # TODO does this include the loss function?
        self.train_fn = self.train_graph.compile()

        self.optimizers = []

    @property
    def trainable_variables(self):
        return list(itertools.chain(*(layer.trainable_variables for layer in self.train_graph.nodes)))

    # def _create_train_op(self):
    #     @tf.function
    #     def train_step(*args):
    #         with tf.GradientTape() as tape:
    #             # predictions = self.train_fn(args)
    #             # TODO modify the train graph with loss output node only
    #             loss = self.loss(*args)  # (labels, predictions)
    #         grads = tape.gradient(loss, self.trainable_variables
    #         # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         # grads = tf.gradients(loss, params)
    #         # for grad, param in zip(grads, params):
    #
    #         # if "clipglobalnorm" in config:
    #         #     grads = [tf.clip_by_global_norm(g, config["clipglobalnorm"]) for g in grads]
    #         #
    #         # if "clipnorm"):
    #         #     grads = [tf.clip_by_norm(g, self.clipnorm) for g in grads]
    #         # if hasattr(self, "clipvalue"):
    #         #     grads = [
    #         #         clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
    #         #         for g in grads
    #         #
    #         #      ]

    def add_optimizer(self, optimizer, **config):
        """
        https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/optimizers/Optimizer

        TODO if we save the optimizer configuration as is, it will convert the ``Param`` callable that returns a
            tensor to a constant value. I should add a way to save this configuration indicating which parameters
            are modifiable

        Optimizer Hyper parameters:
            arguments passed to the optimizer constructor. They can be either regular Python values,
            tensors, or callables. If they are callable, the callable will be called during
            apply_gradients() to get the value for the hyper parameter.
        Args:
            optimizer: optimizer class or instance
            **config: dictionary with parameters for the optimizer, if you want to modify these parameters during
            training pass a ``tx.Param`` instead of a value

        Returns:
            optimizer (Optimizer): configured optimizer instance.
        """
        if isinstance(optimizer, tf.optimizers.Optimizer):
            # attributes can be set on optimizers
            # https://github.com/tensorflow/tensorflow/blob/25006be096cd4f0242a3b979fb212bbd192127b3/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L553
            for attr in config:
                setattr(optimizer, attr, config[attr])
        elif issubclass(optimizer, tf.optimizers.Optimizer):
            optimizer = optimizer.from_config(config)
        else:
            raise TypeError(f"optimizer_cls should be an instance of subclass of tf.optimizers.Optimizer:\n"
                            f"\tgot {type(optimizer)} instead.")

        self.optimizers.append(optimizer)

        return optimizer

    def run(self, data_feed, compiled_graph=False):
        """

        Args:
            data_feed: a dict with {``Layer``:value} or a list of values to be fed to each input
            outputs: [Optional] target outputs to be evaluated (from the run_graph)

        Returns:
            list of value with results of computation for each output in the model

        """
        if data_feed is not None:
            if not isinstance(data_feed, dict):
                inputs = [x for x in self.run_graph.in_nodes if isinstance(x, tx.Input) and not x.constant]
                if isinstance(data_feed, list):
                    logger.warning(
                        "Be careful passing a list to model.run(data), for a model with a single input \"x\"\n"
                        "this is interpreted as feeding x<--data[0], with multiple inputs each item in the list\n"
                        "is fed to the inputs by the same order they are defined x1<--data[0], x2<--data[1], etc")
                data_feed = dict(zip(inputs, data_feed))

            data_feed = {key: data_feed[key]
                         for key in data_feed.keys()
                         if isinstance(key, tx.Layer)}

            # TODO automatic update ops if train has been called

            if not compiled_graph:
                # run eager mode
                return self.run_graph(data_feed)
            else:
                if self.compiled_run is None:
                    self.compiled_run = self.run_graph.compile(ord_inputs=self.run_graph.in_nodes)
                    ord_inputs = [data_feed[k] for k in self.run_graph.in_nodes]

                    return self.compiled_run(*ord_inputs)
