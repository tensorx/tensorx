import tensorflow as tf
import tensorx as tx
from tensorx.utils import as_list, Graph
import itertools
import inspect
import logging
from tensorx.callbacks import *
import numpy as np

logger = logging.getLogger('tensorx')


class Model:
    """
    Args:
        train_loss: either a callable, layer, or dictionary mapping a callable or
            layer to the target outputs.

        train_inputs: defaults to run inputs, if loss is provided you can either
            supply the inputs to the train graph that include the loss, or let
            the Model create inputs for you.
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

        self.update_weights = dict()

    @property
    def trainable_variables(self):
        return list(itertools.chain(*(layer.trainable_variables for layer in self.train_graph.nodes)))

    #
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
    #   - since the loss is a layer it can have a name but these names are not guaranteed to be unique
    #   (I can force this)
    #   - the goal is not to use string names but object so we can have optimizers connected to specific losses
    #   - if multiple losses are defined, setting an optimizer will create a join loss, otherwise a loss must be
    #   specified
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

        TODO in the future optimizer will affect a particular loss if more than one is defined
         or we should have a way to create a combined loss if multiple losses are defined but we
         don't supply a way to handle them

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

        self.optimizer = optimizer
        self.optimizer_params[optimizer] = dict()

        # different optimizers might have the same param names with different values
        for param_name, param in config.items():
            if isinstance(param, tx.Param):
                self.optimizer_params[optimizer][param_name] = param

        if self.train_graph not in self.compiled:
            self.compiled[self.train_graph] = self.train_graph.as_function(ord_inputs=self.train_graph.in_nodes)

        train_fn = self.compiled[self.train_graph]

        @tf.function
        def update_weights(*data):
            with tf.GradientTape() as tape:
                *train_out, loss = train_fn(*data)
                cfg = self.optimizer.get_config()

                grads = tape.gradient(loss, self.trainable_variables)

                if "clipnorm" in cfg:
                    clipnorm = cfg["clipnorm"]
                    grads = [tf.clip_by_norm(g, clipnorm) for g in grads]
                if "clipvalue" in cfg:
                    clipvalue = cfg["clipvalue"]
                    grads = [tf.clip_by_value(g, -clipvalue, clipvalue) for g in grads]

                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

                return train_out + [loss]

        self.update_weights[optimizer] = update_weights

        return optimizer

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
                self.compiled[graph] = graph.as_function(ord_inputs=graph.in_nodes)
                params = list(data_feed.values())
                return self.compiled[graph](*params)

    def run(self, input_feed, compiled_graph=False):
        if input_feed is not None:
            # TODO can models have params to be changed by input feed?
            params = self.model_props
            data_feed, param_feed = Model.parse_input(input_feed, self.run_graph, params)

            # feed all params if necessary
            if param_feed:
                for param_name in param_feed:
                    params[param_name].value = param_feed[param_name]

        return self._run(data_feed, self.run_graph, compiled_graph)

    @staticmethod
    def parse_input(input_feed, graph, param_dict=None):
        """ parses input_feed into data_feed ordered by current graph in_node order and param_feed

        Args:
            input_feed: a dictionary from Layers or string to values, or a list with a value for each
                input in a given graph by the same order these are defined. ``Input`` layer keys map to graph inputs,
                ``str`` keys map to either optimizer parameters or model properties

            graph: the graph used to order the inputs
            param_dict: dict of string to Properties/Params, with params we're trying to match

        Returns:
            data_feed, param_feed, dictionaries the first from ``Input`` layers to values to be fed to these inputs, the
            second from ``str`` values to scalar values to be given to either the current Optimizer or model properties
        """
        if not isinstance(input_feed, dict):
            input_feed = as_list(input_feed)
            inputs = [x for x in graph.in_nodes if isinstance(x, tx.Input) and not x.constant]
            input_feed = dict(zip(inputs, input_feed))

        data_feed = dict()
        param_feed = dict()

        for in_node in input_feed:
            if in_node in graph.in_nodes:
                data_feed[in_node] = input_feed[in_node]
            elif isinstance(in_node, str):
                if (param_dict is not None and in_node in param_dict) or param_dict is None:
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
            params = self.optimizer_params[self.optimizer]
            data_feed, param_feed = Model.parse_input(input_feed, self.train_graph, params)

            # feed all params if necessary
            if param_feed:
                for param_name in param_feed:
                    params[param_name].value = param_feed[param_name]

        optimization_fn = self.update_weights[self.optimizer]
        return optimization_fn(*list(data_feed.values()))

    def train(self, train_data, validation_data=None, test_data=None, epochs=1, steps_per_epoch=None, callbacks=[]):
        """ main training loop

        Args:
            train_data: an iterable of dictionaries from Input Layers to values {Input:data}.
            (calling iter on this object should yield an iterator for an epoch.)
            validation_data: an iterable of dictionaries from Input Layers to values {Input:data}.
            test_data: an iterable of dictionaries from Input Layers to values {Input:data}.
            epochs (int): number of training epochs.
            steps_per_epoch: number of steps in an epoch, if not None, epochs are incremented each time
            this number of steps pass even if the entire train_data has not been transversed.
            callbacks: ``Callback`` functions scheduled during the training.

        Returns:

        """
        # train loop properties
        step = Property("step", 0)
        epoch = Property("epoch", 1)
        epoch_step = Property("epoch_step", 0)
        last_loss = Property("last_loss", 0)
        train_loss = Property("train_loss", None)
        total_epochs = StaticProperty("total_epochs", value=epochs)
        test_loss = Property("test_loss", None)

        # TODO training loop only works with a single optimizer
        #   changing the optimizer during this loop is problematic
        #   without replacing the properties in the scheduler for the new
        #   optimizer
        optimizer_props: dict = self.optimizer_params[self.optimizer]

        properties = [step,
                      epoch_step,
                      epoch,
                      train_loss,
                      last_loss,
                      test_loss,
                      total_epochs
                      ] + list(optimizer_props.values())

        scheduler = Scheduler(model=self,
                              properties=properties)

        for callback in callbacks:
            scheduler.register(callback)

        if steps_per_epoch is not None and train_data is not None:
            epoch_data = iter(train_data)

        if validation_data:
            validation_cb = Eval(target_property="validation_loss",
                                 dataset=validation_data,
                                 priority=-2)
            scheduler.register(validation_cb)

        if test_data:
            test_cb = Eval(property="test_loss",
                           dataset=test_data,
                           priority=-1)
            scheduler.register(test_cb)

        # MAIN TRAINING LOOP
        scheduler.trigger(OnLoop(AT.START))

        try:
            while epoch.value <= epochs:
                # EPOCH START
                # restart iterator for an epoch
                if steps_per_epoch is None and train_data is not None:
                    epoch_data = iter(train_data)

                epoch_step.value = 0
                total_loss = 0
                scheduler.trigger(OnEpoch(epoch.value, AT.START))
                while steps_per_epoch is None or epoch_step.value < steps_per_epoch:
                    try:
                        if train_data is not None:
                            feed_dict = next(epoch_data)
                        else:
                            feed_dict = {}

                        feed_dict, param_feed = Model.parse_input(feed_dict, self.train_graph)

                        optimizer_props = self.optimizer_params[self.optimizer]
                        for param_name in param_feed:
                            if param_name in optimizer_props:
                                optimizer_props[param_name].value = param_feed[param_name]

                        # updated here because we want this property to give us the current step, to know when
                        # a step ends use OnEveryStep(at=AT.END)
                        epoch_step.value += 1
                        step.value += 1
                        # update property values
                        for param_name in param_feed:
                            # add new properties if not in the scheduler
                            if param_name not in scheduler.props:
                                scheduler.observe(Property(name=param_name, value=param_feed[param_name]))
                            else:
                                # only update property if value changes
                                prop = scheduler.props[param_name]
                                # only change the value if this value is different, no need to trigger
                                # redundant updates on the properties that don't change value
                                if prop.value != param_feed[param_name]:
                                    prop.value = param_feed[param_name]

                        scheduler.trigger(OnStep(step.value, AT.START))
                        scheduler.trigger(OnEpochStep(epoch_step.value, AT.START))

                        loss = self.train_step(feed_dict)
                        if not np.isscalar(loss):
                            if isinstance(loss, list):
                                loss = np.mean([np.mean(l) for l in loss])
                            else:
                                loss = np.mean(loss)
                        last_loss.value = loss
                        total_loss += loss
                        train_loss.value = total_loss / epoch_step.value

                        scheduler.trigger(OnStep(epoch_step.value, AT.END))
                        scheduler.trigger(OnEpochStep(epoch_step.value, AT.END))
                    except StopIteration:
                        break

                # EPOCH END
                scheduler.trigger(OnEpoch(epoch.value, AT.END))
                epoch.value += 1
        except StopTrain as e:
            logging.info("Training stopped: {}".format(str(e)))
        except Exception as e:
            logging.exception("Error: " + str(e))
            raise e

        scheduler.trigger(OnLoop(AT.END))


class StopTrain(Exception):
    pass


class Eval(Callback):
    def __init__(self, target_property="eval", eval_fn=None, dataset=None, trigger=OnEveryEpoch(at=AT.END), priority=1):
        """ Evaluation Callback

        Takes a data sample evaluation function and a dataset and returns the average evaluation value for
        the entire dataset.

        Args:
            target_property: name for the property created by this callback
            eval_fn: function applied to the average evaluation value before updating the property
            dataset: the dataset on which the model will be evaluated
            trigger: trigger for when the evaluation is run
            priority: callback priority
        """
        self.target_property = Property(name=target_property)
        self.dataset = dataset
        self.eval_fn = eval_fn

        def eval_fn(model, _):
            dataset_it = iter(self.dataset)
            sum_eval = 0
            steps = 0

            for feed_dict in dataset_it:
                # if feed dict is not a feed dict, it should be a tuple or another iterable
                if not isinstance(feed_dict, dict):
                    inputs = model.eval_graph.input_layers
                    feed_dict = dict(zip(inputs, feed_dict))

                mean_eval = model.eval_step(data_feed=feed_dict)
                if not np.isscalar(mean_eval):
                    mean_eval = np.mean(mean_eval)
                sum_eval += mean_eval
                steps += 1

            avg_eval = sum_eval / steps
            if self.eval_fn:
                self.target_property.value = self.eval_fn(avg_eval)
            else:
                self.target_property.value = avg_eval

        super().__init__(trigger_dict={trigger: eval_fn},
                         properties=[self.target_property],
                         priority=priority)
