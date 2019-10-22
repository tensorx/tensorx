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
        self.compiled = dict()

        if not isinstance(train_loss, tx.Layer):
            raise TypeError(f"Invalid train_loss type\n"
                            f"\t expected: Layer"
                            f"\t actual: {type(train_loss)}")

        self.train_loss = train_loss

        self.train_graph: Graph = Graph.build(self.train_inputs, self.train_outputs + [self.train_loss])

        self.eval_graph: Graph = Graph.build(self.eval_inputs, self.eval_outputs)

        self.optimizer = None

        # optimizer: Param:
        self.optimizer_params = dict()
        # model properties accessible to callbacks
        self.model_props = set()

        self.update_model = dict()

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
    # TODO I wonder if I can get just one optimizer per model
    #   the use case would be train a model n steps with an optimizer then use a callback to switch
    #   to a different optimizer etc.
    #   In this case the train loop would still be a single loop but the optimizers would switch inside of it
    #   I would need the option to switch optimizers manually
    #   but I still have a single loss
    #   Use case 2.
    #   -how to switch between two losses ?
    #   -I guess different losses would use different optimization steps since we need to gather the gradients
    #    for them
    #   - in keras you set the loss with compile, but where it connects along with its inputs becomes fuzzy
    #   - unless we have a way to set the optimizer to a specific loss?
    #   - since the loss is a layer it can have a name but these names are not guaranteed to be unique (I can force this)
    #   - the goal is not to use string names but object so we can have optimizers connected to specific losses
    #   - if multiple losses are defined, setting an optimizer will create a join loss, otherwise a loss must be specified
    #   - the model subclass should be responsible for exposing names for the loss references if necessary
    #   SET OPTIMIZER
    #   1. if no loss specified, create join loss with MEAN layer
    #   2. if loss specified create optimize step for that loss
    #   3. train will call specific optimizer or all of them in sequence? for the same data? this is a problem
    #   I guess I can ignore the later feature for now
    def set_optimizer(self, optimizer, **config):
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

        # TODO params have to be bound to optimizers after all

        self.optimizer = optimizer
        self.optimizer_params[optimizer] = dict()

        # different optimizers might have the same param names with different values
        for param_name, param in config.items():
            if isinstance(param, tx.Param):
                self.optimizer_params[optimizer][param_name] = param

        # TODO in the future optimizer will affect a particular loss if more than one is defined
        #   or we should have a way to create a combined loss if multiple losses are defined but we
        #   don't supply a way to handle them

        # TODO originally the train loss was defined as
        #  train_loss: either a callable, layer, or dictionary mapping a callable or layer to the target outputs
        # train_outputs
        # loss (applied to train outputs?) or as a layer that has nodes in the train graph
        # train_inputs should be defined or not, in any case, loss will have access to some inputs

        # TODO a way to turn this into a layer is to allow us to set the values directly into the inputs
        if self.train_graph not in self.compiled:
            self.compiled[self.train_graph] = self.train_graph.compile(ord_inputs=self.train_graph.in_nodes)

        train_fn = self.compiled[self.train_graph]

        def update_weights(*data):
            with tf.GradientTape() as tape:
                train_out, loss = train_fn(*data)
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return loss

        self.update_model[optimizer] = update_weights

        return optimizer

    def parse_input(self, input_feed, graph, optimizer):
        """ parses input_feed into data_feed ordered by current graph in_node order and param_feed

        Args:
            input_feed: a dictionary from Layers or string to values, or a list with a value for each
                input in a given graph by the same order these are defined. ``Input`` layer keys map to graph inputs,
                ``str`` keys map to either optimizer parameters or model properties

            graph: the graph used to order the inputs
            optimizer: the optimizer for which the parameters we're trying to match

        Returns:
            data_feed, param_feed, dictionaries the first from ``Input`` layers to values to be fed to these inputs, the
            second from ``str`` values to scalar values to be given to either the current Optimizer or model properties
        """
        if not isinstance(input_feed, dict):
            inputs = [x for x in graph.in_nodes if isinstance(x, tx.Input) and not x.constant]
            input_feed = dict(zip(inputs, input_feed))

        data_feed = dict()
        param_feed = dict()

        for in_node in input_feed:
            if in_node in graph.in_nodes:
                data_feed[in_node] = input_feed[in_node]
            elif isinstance(in_node, str) and in_node in self.model_props:
                param_feed[in_node] = input_feed[in_node]
            elif isinstance(in_node, str) and in_node in self.optimizer_params[optimizer]:
                param_feed[in_node] = input_feed[in_node]
            else:
                if not isinstance(in_node, (str, tx.Input)):
                    raise TypeError(f"invalid type fed to model {type(in_node)}:\n"
                                    f"\texpected either Input Layer or str")
                else:
                    raise ValueError(f"{str(in_node)} not found neither in graph nor params")

        if len(data_feed) != len(graph.in_nodes):
            raise ValueError(f"model required {len(graph.in_nodes)} inputs, {len(data_feed)} found")

        # order data_feed
        data_feed = {in_node: data_feed[in_node] for in_node in graph.in_nodes}

        return data_feed, param_feed

    def train_step(self, input_feed):
        if input_feed is not None:
            data_feed, param_feed = self.parse_input(input_feed, self.train_graph, self.optimizer)

            # feed all params if necessary
            if param_feed:
                for param in param_feed:
                    param.value = param_feed[param]

        optimization_fn = self.update_model[self.optimizer]
        return optimization_fn(*list(data_feed.values()))

    def _run(self, data_feed, graph: Graph, compiled_graph=False):
        """
            Args:
                data_feed: a dict with {``Layer``:value} or a list of values to be fed to each input

            Returns:
                list of value with results of computation for each output in the model

        """
        if not compiled_graph:
            # run eager mode
            # print(data_feed)
            return graph(data_feed)
        else:
            if graph not in self.compiled:
                self.compiled[graph] = graph.compile(ord_inputs=graph.in_nodes)
                params = list(data_feed.values())
                return self.compiled[graph](*params)

    def run(self, input_feed, compiled_graph=False):
        if input_feed is not None:
            data_feed, param_feed = self.parse_input(input_feed, self.run_graph, self.optimizer)

            # feed all params if necessary
            if param_feed:
                for param in param_feed:
                    param.value = param_feed[param]

        return self._run(data_feed, self.run_graph, compiled_graph)
