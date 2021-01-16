import tensorflow as tf
import tensorx as tx
from tensorx.utils import Graph
import logging
from tensorx.train.callbacks import *
import numpy as np
from typing import Union, Dict
from queue import Empty
import os
import csv
import re

logger = logging.getLogger('tensorx')


# TODO convert Callable train_loss to layer
class Model:
    """ Base Model
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
                 eval_score=None,
                 name='Model',
                 ):
        self.name = name

        self.run_inputs = as_list(run_inputs)
        self.run_outputs = as_list(run_outputs)
        self.train_inputs = as_list(train_inputs)
        self.train_outputs = as_list(train_outputs)
        self.eval_inputs = as_list(eval_inputs)
        self.eval_outputs = as_list(eval_outputs)
        self.eval_score = as_list(eval_score)

        if not isinstance(train_loss, tx.Layer):
            raise TypeError(f"Invalid train_loss type\n"
                            f"\t expected: Layer"
                            f"\t actual: {type(train_loss)}")

        self.train_loss = train_loss

        # layer graph -> function
        self.compiled = dict()

        # TODO problem with missing inputs in recurrent neural networks, solved by allowing missing inputs
        #  and supplying the inputs when doing as_function
        self.run_graph: Graph = Graph.build(inputs=run_inputs,
                                            outputs=run_outputs,
                                            add_missing_inputs=True)
        self.train_graph: Graph = Graph.build(inputs=self.train_inputs,
                                              outputs=self.train_outputs + [self.train_loss],
                                              add_missing_inputs=True)
        self.eval_graph: Graph = Graph.build(inputs=self.eval_inputs,
                                             outputs=self.eval_outputs + self.eval_score,
                                             add_missing_inputs=True)

        self.graph_inputs = {self.run_graph: self.run_inputs,
                             self.train_graph: self.train_inputs,
                             self.eval_graph: self.eval_inputs}

        # TODO add the possibility of having multiple optimizers that can be switched
        self.optimizer = None

        # optimizer: Param:
        self.optimizer_params = dict()
        # model properties accessible to callbacks
        self.optimization_step = dict()

        self.model_props = set()

    def draw(self, path="graph.pdf"):
        # TODO add edges for shared state
        try:
            from pygraphviz import AGraph

            def add_graph(g: AGraph, layer_graph: tx.Graph, cluster):
                for node in layer_graph.nodes:
                    # HTML for record nodes https://graphviz.org/doc/info/shapes.html#top
                    g.add_node(f"{cluster}_{node.name}", shape="none", margin=0, label=tx.utils.vizstyle(node))
                for node in layer_graph.nodes:
                    for other_node in layer_graph.edges_out[node]:
                        g.add_edge(f"{cluster}_{node.name}", f"{cluster}_{other_node.name}")

            dg = AGraph(directed=True)
            if self.run_graph and self.run_graph.nodes:
                dg.add_subgraph(name="cluster_run", label="run")
                g = dg.subgraphs()[-1]
                add_graph(g, self.run_graph, cluster="run")
            if self.train_graph and self.train_graph.nodes:
                dg.add_subgraph(name="cluster_train", label="train")
                g = dg.subgraphs()[-1]
                add_graph(g, self.train_graph, "train")
            if self.eval_graph and self.eval_graph.nodes:
                dg.add_subgraph(name="cluster_eval", label="eval")
                g = dg.subgraphs()[-1]
                add_graph(g, self.eval_graph, cluster="eval")

            dg.layout(prog="dot")
            dg.draw(path=path)

        except ImportError:
            raise ImportError("Could't find required pygraphviz module")

    @property
    def trainable_variables(self):
        return list(itertools.chain(*(layer.trainable_variables for layer in self.train_graph.nodes)))

    def set_optimizer(self, optimizer, **config):
        """ Set the optimizer for this model

        !!! Note "Optimizer Hyper-Parameters"
            The arguments passed to the optimizer constructor can be either regular Python values,
            tensors, or a `callable`. If they are callable, they will called during apply_gradients()
            to get the value for the hyper parameter.

        Args:
            optimizer (Optimizer): optimizer class or instance
            **config : dictionary with parameters for the optimizer, if you want to modify these parameters during
            training pass an ``tx.Param`` as the value for the given parameter instead of constant value.

        Returns:
            optimizer (Optimizer): the configured optimizer instance.
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
            self.compiled[self.train_graph] = self.train_graph.as_function(ord_inputs=self.train_inputs)
        train_fn = self.compiled[self.train_graph]

        @tf.function
        def optimization_step(*data):
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

        self.optimization_step[optimizer] = optimization_step

        return optimizer

    def run(self, input_feed, compiled_graph=False):
        if input_feed is not None:
            # TODO can models have params to be changed by input feed?
            params = self.model_props
            data_feed, param_feed = Model.parse_input(input_feed, self.run_inputs, params)

            # feed all params if necessary
            if param_feed:
                for param_name in param_feed:
                    params[param_name].value = param_feed[param_name]

        if not compiled_graph:
            return self.run_graph(data_feed)
        else:
            if self.run_graph not in self.compiled:
                self.compiled[self.run_graph] = self.run_graph.as_function(ord_inputs=self.run_inputs)

        params = list(data_feed.values())
        return self.compiled[self.run_graph](*params)

    @staticmethod
    def parse_input(input_feed, ord_inputs, param_dict=None):
        # TODO this has to be model dependent? or receive the input nodes for the current model instead of graph
        #  because the only thing we take from graph is in_nodes
        """ parses input_feed into data_feed ordered by current graph in_node order and param_feed

        Args:
            input_feed: a dictionary from Layers or string to values, or a list with a value for each
                input in a given graph by the same order these are defined. ``Input`` layer keys map to graph inputs,
                ``str`` keys map to either optimizer parameters or model properties

            ord_inputs: the ordered inputs to be matched with input_feed dict or list order
            param_dict: dict of string to Properties/Params, with params we're matching with input_feed keys

        Returns:
            data_feed, param_feed, dictionaries the first from ``Input`` layers to values to be fed to these inputs, the
            second from ``str`` values to scalar values to be given to either the current Optimizer or model properties
        """
        if not isinstance(input_feed, dict):
            input_feed = as_list(input_feed)
            inputs = [x for x in ord_inputs if isinstance(x, tx.Input) and not x.constant]
            input_feed = dict(zip(inputs, input_feed))

        data_feed = dict()
        param_feed = dict()

        for in_node in input_feed:
            if in_node in ord_inputs:
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

        if len(data_feed) != len(ord_inputs):
            raise ValueError(f"model required {len(ord_inputs)} inputs, {len(data_feed)} found")

        # order data_feed
        data_feed = {in_node: data_feed[in_node] for in_node in ord_inputs}

        return data_feed, param_feed

    def train_step(self, input_feed):
        if input_feed is not None:
            params = self.optimizer_params[self.optimizer]
            data_feed, param_feed = Model.parse_input(input_feed, self.train_inputs, params)

            # feed all params if necessary
            if param_feed:
                for param_name in param_feed:
                    params[param_name].value = param_feed[param_name]

            if data_feed:
                for lr in data_feed:
                    if isinstance(lr, tx.Input) and not lr.constant:
                        lr.value = data_feed[lr]

        optimization_step = self.optimization_step[self.optimizer]
        feed_values = list(data_feed.values())
        return optimization_step(*feed_values)

    def eval_step(self, input_feed):
        """

        Args:
            input_feed:

        Returns:
            *eval_output, eval_score ((eval outputs,eval score)):
        """
        if input_feed is not None:
            data_feed, param_feed = Model.parse_input(input_feed, self.eval_inputs)

            eval_graph = self.eval_graph
            if eval_graph not in self.compiled:
                self.compiled[eval_graph] = eval_graph.as_function(ord_inputs=self.eval_inputs, compile=True)

            static_eval_graph = self.compiled[eval_graph]
            feed_values = list(data_feed.values())
            return static_eval_graph(*feed_values)
        else:
            return None

    def train(self, train_data, validation_data=None, test_data=None, epochs=1, steps_per_epoch=None, callbacks=[]):
        """ Main training loop

        Args:
            train_data: an iterable of dictionaries from Input Layers to values {Input:data}.
            (calling iter on this object should yield an iterator for an epoch.)

            validation_data: an iterable of dictionaries from Input Layers to values {Input:data}.
            test_data: an iterable of dictionaries from Input Layers to values {Input:data}.

            epochs (int): number of training epochs.
            steps_per_epoch: number of steps in an epoch, if not None, epochs are incremented each time
            this number of steps pass even if the entire train_data has not been transversed.

            callbacks: ``Callback`` functions scheduled during the training.

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

                        feed_dict, param_feed = Model.parse_input(feed_dict, self.train_inputs)

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

                        *outputs, loss = self.train_step(feed_dict)
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
                    inputs = model.eval_graph.in_nodes
                    feed_dict = dict(zip(inputs, feed_dict))

                # *outputs, score
                eval_output = model.eval_step(feed_dict)
                *_, mean_eval = eval_output
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


class EarlyStop(Callback):
    """ EarlyStop Callback

    Executes each time a given target property changes

    if the callback depends on another callback that computes a given property, make sure EarlyStop
    is executed after this callback.

    Warning:
        if this relies on a property doesn't exist, it throws an error and interrupts the train loop.


    """

    def __init__(self,
                 patience=3,
                 lesser_better=True,
                 threshold=0.001,
                 target="validation_loss",
                 trigger=None, priority=1):
        self.patience = patience
        self.threshold = threshold,
        self.target = target
        self.last_eval = None
        self.patience_tick = 0

        def early_stop(model, properties):
            if self.target not in properties:
                raise KeyError(
                    "EarlyStop callback is trying to access a property that doesn't exist: {}".format(self.target))
            measure = properties[target].value
            if self.last_eval is None:
                self.last_eval = measure
            else:
                improvement = self.last_eval - measure
                if not lesser_better:
                    improvement = -1 * improvement

                if improvement < self.threshold:
                    self.patience_tick += 1
                    if self.patience_tick == self.patience:
                        raise StopTrain("the model didn't improve for the last {} epochs".format(patience))
                else:
                    self.patience_tick = 0

                self.last_eval = measure

        if trigger is None:
            trigger = OnValueChange(self.target)

        super().__init__(trigger_dict={trigger: early_stop}, priority=priority)


class StopOnNaN(Callback):
    """ StopOnNaN Callback

    Interrupts the training loop if the last_loss property returns NaN

    Note:
        the only event triggered after this is ``OnTrain(AT.END)``
    """

    def __init__(self):
        def raise_stop(model, properties):
            step = properties["epoch_step"].value
            epoch = properties["epoch"].value
            loss = properties["last_loss"].value
            if np.isnan(loss):
                raise StopTrain("loss returned NaN on epoch {epoch} step {step}".format(epoch=epoch, step=step))

        super().__init__({OnValueChange("last_loss"): raise_stop})


class Progress(Callback):
    """ Progress Callback

    creates a CLI progress bar for the training loop
    """

    def __init__(self, total_steps=None, monitor="train_loss", priority=1):
        self.total_steps = total_steps
        self.progress = None
        self.monitor = as_list(monitor)

        # optional, only tries to import it when we create a progress callback

        from tqdm.autonotebook import tqdm
        self.tqdm = tqdm

        def progress_init(model, properties):
            self.progress = self.tqdm(total=self.total_steps)

        def progress_step(model, properties):
            postfix = {}
            for prop_name in self.monitor:
                if prop_name not in properties:
                    raise KeyError(
                        "Progress callback is trying to access a property that doesn't exist: {}".format(self.monitor))
                prop = properties[prop_name].value
                postfix[prop_name] = prop
            self.progress.update(1)
            if len(postfix) > 0:
                self.progress.set_postfix(postfix)

        def progress_stop(model, properties):
            self.progress.close()

        trigger_dict = {OnLoop(AT.START): progress_init,
                        OnEveryStep(at=AT.END): progress_step,
                        OnLoop(AT.END): progress_stop}

        def progress_update(model, properties):
            if self.total_steps is None:
                self.total_steps = self.progress.n * properties["total_epochs"].value
                self.progress.total = self.total_steps

        if self.total_steps is None:
            trigger_dict[OnEveryEpoch(at=AT.END)] = progress_update

        super().__init__(trigger_dict=trigger_dict,
                         priority=priority)


class NewProperty(Callback):
    """ NewProperty Callback

    Creates a new property by applying a given function to the value of an existing property.

    """

    def __init__(self, target, fn, new_prop, init_val=None, triggers=OnEveryStep(), priority=1):
        self.property = Property(name=new_prop, value=init_val)
        self.fn = fn
        self.triggers = as_list(triggers)

        def update_prop(model, properties):
            if target not in properties:
                raise ValueError("Property callback tried to access property that doesn't exist: {}".format(target))
            prop = properties[target]
            self.property.value = self.fn(prop.value)

        trigger_dict = {trigger: update_prop for trigger in self.triggers}

        super().__init__(trigger_dict=trigger_dict, priority=priority, properties=self.property)


class LambdaCallback(Callback):
    """ Lambda Callback

    Executes a given function on a given trigger

    """

    def __init__(self, fn=None, triggers=OnEveryEpoch(at=AT.END), properties=[], priority=1):
        self.triggers = as_list(triggers)
        self.properties = properties
        self.fn = fn
        trigger_dict = {trigger: fn for trigger in triggers}

        super().__init__(trigger_dict=trigger_dict, priority=priority, properties=self.properties)


class DictLogger(Callback):
    """ DictLogger Callback

    Logs the values of the given properties to a dictionary accessible through the
    ``logs` attribute

    Args:
        props (List[str]): list of strings with names of properties to log
        trigger (callbacks.Event): even on which this callback is executed
        priority (int): callback priority (lower values have higher priority)

    Attributes:
        logs (dict): dictionary mapping property names (str) to lists of values
    """

    def __init__(self, props, trigger=OnEveryEpoch(at=AT.END), priority=1):
        self.props = as_list(props)
        self.logs = {name: [] for name in self.props}

        def get_props(model, properties):
            for prop_name in self.props:
                if prop_name not in properties:
                    raise KeyError("DictLogger tried to access a property that doesn't exist: {}".format(prop_name))
                prop = properties[prop_name]
                self.logs[prop_name].append(prop.value)

        super().__init__(trigger_dict={trigger: get_props}, priority=priority)


class DecayAfter(Callback):
    """ DecayAfter Callback

    Decays a given property using a given decay rate after a given number of epochs. This callback checks the model
    "epoch" property set in the main training loop, if epoch is greater or equal than ``decay_after`` then it decays
    a target property value using decay_rate.

    Args:
        decay_after (int): epoch after which the decay starts affecting a target property
        target_property (str): a name for the property to be changed
        decay_rate (float): the rate by which the property is changed with `value * decay_rate``
        decay_threshold (float): minimum value the target property will take after which the decay has no effect.
        priority (int) callback priority (lower values have higher priority)

    """

    def __init__(self,
                 decay_after=0,
                 target_property="learning_rate",
                 decay_rate=1.0,
                 decay_threshold=1e-6,
                 priority=1):
        self.target_property = target_property
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.decay_after = decay_after

        def update_fn(_, properties):
            if self.target_property not in properties:
                raise KeyError(
                    "DecayAfter callback is trying to change a property that doesn't exist: {}".format(
                        self.target_property))
            epoch = properties["epoch"].value

            if epoch >= self.decay_after:
                prop = properties[self.target_property]
                prop.value = max(self.decay_threshold, prop.value * decay_rate)

        super().__init__(trigger_dict={OnEveryEpoch(at=AT.END): update_fn},
                         properties=[],
                         priority=priority)


class PlateauDecay(Callback):
    """ PlateauDecay Callback

    Decays a ``target`` property with a given ``decay_rate`` when a ``monitor`` property value plateaus: doesn't improve
    from the previously observed value.

    Note: patience param is reset after decay is applied

    Args:
        monitor (str): the measure on which we want to measure the improvement, by default "validation_loss"
        target (str): the property to be changed by this callback
        decay_threshold (float):value representing the difference between evaluations necessary for the update to occur
        decay_rate (float): rate through which the param value is reduced `(value = value * decay_rate)`
        decay_threshold: point beyond witch the param value is not reduced `max(value * decay_rate, decay_threshold)`
        less_is_better: if True, evaluation is considered to improve if it decreases, else it is considered to improve
        if it increases

    Attributes:
        eval_history (list): list with the evaluation values passed through the update function
        improvement_threshold (float): the necessary difference between evaluations for the update to occur
        decay_rate (float): rate through which the param value is reduced `(value = value * decay_rate)`
        decay_threshold (float): point beyond witch ``target`` is not reduced `max(value * decay_rate, decay_threshold)`
    """

    def __init__(self,
                 monitor,
                 target,
                 improvement_threshold=1.0,
                 less_is_better=True,
                 patience=1,
                 decay_rate=1.0,
                 decay_threshold=1e-6,
                 priority=1):
        self.improvement_threshold = improvement_threshold
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.less_is_better = less_is_better
        self.monitor = monitor
        self.target = target
        self.eval_history = []
        self.patience_tick = 0
        self.patience = patience

        def update_fn(_, properties):
            # get properties
            evaluation = properties[self.monitor]
            to_change = properties[self.target]

            self.eval_history.append(evaluation.value)

            if len(self.eval_history) > 1:
                evaluation = 0
                if len(self.eval_history) > 1:
                    evaluation = self.eval_history[(-2 - self.patience_tick)] - self.eval_history[-1]
                    if not self.less_is_better:
                        evaluation = -1 * evaluation

                if evaluation < self.improvement_threshold:
                    self.patience_tick += 1
                    if self.patience_tick == self.patience:
                        to_change.value = max(to_change.value * self.decay_rate, self.decay_threshold)
                        self.patience_tick = 0
                else:
                    self.patience_tick = 0

        super().__init__(trigger_dict={OnEveryEpoch(at=AT.END): update_fn},
                         properties=[],
                         priority=priority)


class CSVLogger(Callback):
    """ CSVLogger Callback

    logs property values to a csv file.

    Args:
        monitors (List[str]): list of property names to be logged
        static_logs (dict): a dictionary of values (str to value) to be output along with the target properties
    """

    def __init__(self, monitors, out_filename, static_logs=None, trigger=OnEveryEpoch(at=AT.END), priority=1):
        self.property_names = as_list(monitors)
        self.static_logs = static_logs
        self.out_filename = out_filename
        self.out_file = None
        self.n = 0
        self.writer = None

        def get_props(props):
            props = {prop_name: props[prop_name] for prop_name in self.property_names}
            all_props = {p.name: p.value for p in props.values()}
            # add values from the provided logs
            all_props.update(self.static_logs)
            return all_props

        def log_init(model, properties):
            self.out_file = open(self.out_filename, "w")
            all_props = get_props(properties)
            self.writer = csv.DictWriter(f=self.out_file, fieldnames=all_props.keys())
            self.writer.writeheader()

        def log_clean(model, properties):
            self.out_file.close()

        def log(model, properties):
            all_props = get_props(properties)
            self.writer.writerow(all_props)
            self.out_file.flush()
            self.n += 1

        super().__init__(trigger_dict={OnStep(1, at=AT.START): log_init,
                                       trigger: log,
                                       OnLoop(at=AT.END): log_clean},
                         properties=None,
                         priority=priority)


class ResetState(LambdaCallback):
    """ ResetState Callback

    calls reset state on the model on the given triggers
    """

    def __init__(self,
                 triggers=OnEveryEpoch(at=AT.END),
                 priority=1):
        triggers = as_list(triggers)

        def reset(model: Model, _):
            model.reset_state()

        super().__init__(fn=reset, triggers=triggers, priority=priority)


class Plot(Callback):
    """ Callback that plots the given properties in real time

    """

    # noinspection PyBroadException
    def __init__(self, monitor,
                 cols=3,
                 backend='pyqtgraph',
                 keep_open=False,
                 save_plot=False,
                 output_file=None,
                 trigger=OnEveryEpoch(at=AT.END), priority=1):
        self.keep_open = keep_open
        self.monitor = set(as_list(monitor))
        self.output_file = output_file
        self.process = None
        self.queue = None
        self.stop_event = None
        self.backend = backend

        def plot_worker_mpl(queue, stop_event):
            out_file = self.output_file

            step = 1
            axs = {}

            num_props = len(self.monitor)
            rows = np.ceil(num_props / 3)

            if backend == "matplotlib":
                import matplotlib.pyplot as plt
                # only use the TkAgg backend if in console
                try:
                    from IPython import get_ipython
                    if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
                        raise ImportError("console")
                except Exception:
                    import matplotlib
                    matplotlib.use("TkAgg")

                plt.ion()
                fig = plt.figure()
                fig.canvas.toolbar.pack_forget()

                for i, prop_name in enumerate(self.monitor):
                    axs[prop_name] = fig.add_subplot(rows, cols, i + 1)  # i + 1)

            elif backend == "pyqtgraph":
                from pyqtgraph.Qt import QtGui
                import pyqtgraph as pg

                # pg.setConfigOption('background', 'w')
                app = QtGui.QApplication([])
                win = pg.GraphicsWindow(title="Properties")
                plots = {}
                text = {}
                # win.ci.setContentsMargins(20, 0, 0, 0)

                for i, prop_name in enumerate(self.monitor):
                    plot_item = win.addPlot(title=prop_name, row=i // cols, col=i % cols)
                    plots[prop_name] = plot_item
                    axs[prop_name] = plot_item.plot(pen=pg.mkPen('#FF2400', width=1))

                QtGui.QApplication.processEvents()
            else:
                raise ValueError("invalid backed, valid backend options: matplotlib, pyqtgraph")

            while True:
                try:
                    properties = queue.get_nowait()

                    for prop_name in properties.keys():
                        prop_value = properties[prop_name]
                        ax = axs[prop_name]

                        # if prop_value is None:
                        #    prop_value = 0

                        if self.backend == "matplotlib":
                            if step == 1:
                                xs = [step]
                                ys = [prop_value]
                            else:
                                line = ax.lines[0]
                                xs = np.append(line.get_xdata(), [step])
                                ys = np.append(line.get_ydata(), [prop_value])

                                line.set_xdata(xs)
                                line.set_ydata(ys)

                            ax.plot(xs[step - 1:], ys[step - 1:], color="#FF2400", linestyle="solid")

                            if step == 1:
                                plt.tight_layout()
                                ax.set_title(prop_name)
                        elif self.backend == "pyqtgraph":
                            xs, ys = ax.getData()
                            if prop_value is not None:
                                if xs is None:
                                    xs = []
                                    ys = []
                                elif len(xs) == 1:
                                    # create text label
                                    curve_point = pg.CurvePoint(ax)
                                    plots[prop_name].addItem(curve_point)
                                    label = pg.TextItem("test", color="#FF4200")
                                    label.setParentItem(curve_point)
                                    text[prop_name] = (curve_point, label)

                                xs = np.append(xs, [step])
                                ys = np.append(ys, [prop_value])

                                ax.setData(xs, ys)

                                if len(xs) > 1:
                                    # update text label
                                    curve_point, label = text[prop_name]
                                    curve_point: pg.CurvePoint = curve_point
                                    # curve point index relative to sample index (last in this case)
                                    curve_point.setIndex(step - 1)
                                    # anchor is relative to parent (top left corner)
                                    label.setAnchor((1.0, 1.0))
                                    label.setText('%0.3f' % (ys[-1]))

                    if self.backend == "matplotlib":
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    else:
                        QtGui.QApplication.processEvents()

                    step += 1
                except Empty:
                    if stop_event.is_set():
                        break
                    else:
                        if self.backend == "pyqtgraph":
                            QtGui.QApplication.processEvents()

            if self.backend == "matplotlib":
                plt.ioff()
                if self.keep_open:
                    plt.show()
                if save_plot:
                    out_file = self.output_file if self.output_file is not None else "train_run_{}.pdf".format(
                        str(os.getpid()))

                    plt.savefig(out_file)
                plt.close(fig)
            else:

                if save_plot:
                    import pyqtgraph.exporters
                    exporter = pg.exporters.ImageExporter(win.scene())
                    if out_file is None:
                        out_file = "train_run_{}.pdf".format(str(os.getpid()))
                    exporter.export(out_file)
                if self.keep_open:
                    QtGui.QApplication.instance().exec_()
                app.closeAllWindows()

        def plot_init(model, properties):
            import multiprocessing as mp
            self.stop_event = mp.Event()
            self.queue = mp.Queue()
            self.process = mp.Process(target=plot_worker_mpl, args=(self.queue, self.stop_event,))
            self.process.start()

        def plot_props(model, properties):
            data = {}
            for prop_name in self.monitor:
                if prop_name not in properties:
                    raise KeyError("Plot callback tried to access a property that doesn't exist: {}".format(prop_name))

                y = properties[prop_name].value
                data[prop_name] = y
            self.queue.put(data)

        def plot_clean(model, properties):
            self.stop_event.set()
            self.process.join()

        super().__init__(trigger_dict={OnLoop(at=AT.START): plot_init,
                                       trigger: plot_props,
                                       OnLoop(at=AT.END): plot_clean}, priority=priority)
