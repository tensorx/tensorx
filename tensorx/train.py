"""  Training Module

Contains Module containers to wrap and train neural network models

Contains Learners which are simplified versions of Optimizers. Instead of working with
gradients, they work with delta values which are not necessarily the result of an optimization
process (minimization of a loss function)

This module contains learning procedures different from loss functions used
with gradient descend methods such Winner-Takes-All (WTA) methods for Self-Organising Maps
"""

import os
from pygraphviz import AGraph
from abc import ABCMeta, abstractmethod

from tensorx.layers import DynamicParam
from tensorx.utils import as_list
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.client.session import Session, InteractiveSession
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import array_ops, math_ops, control_flow_ops
from tensorflow.python.ops.gen_state_ops import scatter_sub
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.variables import Variable
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training.saver import Saver, import_meta_graph, export_meta_graph
from tensorflow.python.summary import summary

from tensorflow.core.protobuf.config_pb2 import RunOptions, RunMetadata
from tensorx.layers import *
from tensorx.utils import Graph


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


def _get_feedable(inputs):
    feedable = []
    for elem in inputs:
        if hasattr(elem, 'placeholder'):
            feedable.append(elem)
    return feedable


class LayerGraph:
    def __init__(self, outputs, inputs=None, other_tensors=None, other_inputs=None):
        """ Creates a self-contained graph that can be evaluated by feeding data to the
        graph inputs.

        Args:
            other_inputs (object): other inputs are just added to the graph possibly
            to feed other tensors graph, but depdendencies are not checked.
            other_tensors (List[Tensor]): if supplied these are runned along with the rest of the graph in eval
            AFTER the fetches from the outputs
            inputs: a list of lists of inputs (a Input, SparseInput Layers or Param instance) for each output node

            outputs: a list of Layers with the outputs of the graphs

        Raises:
            Value Error: if the graph is not well defined: the outputs are not connected to the
            specified inputs
        """
        self.other_tensors = as_list(other_tensors)
        self.other_inputs = as_list(other_inputs)
        self.input_layers = set(as_list(inputs))
        self.output_layers = as_list(outputs)

        dependencies = {}
        graph = Graph()
        for output in self.output_layers:
            dependencies[output] = []

        """
        def build_graph(current_layer, origin_layer, to_visit):
            in_layers = current_layer.input_layers
            if len(in_layers) == 0 and hasattr(current_layer, "placeholder"):
                if current_layer not in dependencies[origin_layer]:
                    dependencies[origin_layer].append(current_layer)
            else:
                for in_layer in in_layers:
                    graph.add_edge(in_layer, current_layer)
                    if in_layer not in visited:
                        build_graph(in_layer, origin_layer)
        """
        visited = set()

        def build_graph(output_layer):
            to_visit = [output_layer]
            while to_visit:
                current_layer = to_visit.pop(0)
                if current_layer not in visited:
                    visited.add(current_layer)
                    if len(current_layer.input_layers) == 0 and hasattr(current_layer, "placeholder"):
                        if current_layer not in dependencies[output_layer]:
                            dependencies[output_layer].append(current_layer)
                    for input_layer in current_layer.input_layers:
                        graph.add_edge(input_layer, current_layer)
                        if input_layer not in visited and input_layer not in to_visit:
                            to_visit.append(input_layer)

        missing_dep = {}
        for output in self.output_layers:
            build_graph(output)
            for dependency in dependencies[output]:
                if dependency not in self.input_layers and len(self.input_layers) != 0:
                    if output not in missing_dep:
                        missing_dep[output] = []
                    if dependency not in missing_dep[output]:
                        missing_dep[output].append(dependency)

        if len(missing_dep) > 0:
            raise ValueError("Could not create graph: \n Missing input dependencies:"
                             "\n {missing}".format(
                missing="\n".join([str(o) + "<---[{}]".format(",".join(map(str, i))) for o, i in missing_dep.items()])))

        self.graph = graph
        self.dependencies = dependencies
        self.layers = visited

        all_dependencies = set()
        for dep in self.dependencies.values():
            all_dependencies.update(dep)

        unused_dependencies = set(self.input_layers).difference(all_dependencies)
        if len(unused_dependencies) > 0:
            missing_str = "\n".join(map(str, unused_dependencies))
            raise ValueError("One of the Inputs is not a dependency of any Output. \n"
                             "Unused inputs: \n {unused}".format(unused=missing_str))

        if len(self.input_layers) == 0:
            self.input_layers = list(all_dependencies)

    def missing_dependencies(self, inputs, outputs=None) -> dict:
        """ given a list of input layers, checks if any input dependency is missing from the graph

        Args:
            outputs: the outputs for which we want the missing input dependencies
            inputs: a list of input layers that are dependencies to output nodes in this graph

        Returns:
            a dictionary with missing dependencies for each output in this graph

        """
        if outputs is None:
            outputs = self.output_layers
        missing_dep = {}
        for output in outputs:
            for dep in self.dependencies[output]:
                if dep not in inputs:
                    if output not in missing_dep:
                        missing_dep[output] = set()
                    missing_dep[output].add(dep)
        return missing_dep

    def eval(self, feed=None, other_tensors=None, target_outputs=None,
             use_defaults=True, session=None, options=None,
             run_metadata=None):
        """ Evaluates the current graph on the given inputs

        if input_values are used and Inputs have values != None, these are not overwritten
        if a feed dictionary with layer-->data is passed, only the missing inputs are possibly
        fed with their default values.

        Args:
            other_tensors: runs other tensors or ops that might not be included in the graph
            use_defaults: automatically fill the default values if input layer .value attribute is not None
            input_values: data to be fed, else it tries to match the input values with all self.input_lauyers
            feed: a feed dictionary from Input Layers or Parameters to values, if None, matches the
            inputs with self.inputs in the same order. If default values are available in the input
            layers, these are used instead.
            session: a session to be used to evaluate this graph, if None, uses the default session

            options: A [RunOptions] protocol buffer
            run_metadata: A [RunMetadata] protocol buffer

        Returns:
            the result of the graph evaluation

        """
        if not isinstance(feed, dict):
            raise TypeError("feed must be a dictionary from inputs to values")

        if session is None:
            session: Session = ops.get_default_session()

        target_outputs = as_list(target_outputs)

        output_layers = self.output_layers
        # input_layers = self.input_layers
        other_tensors = self.other_tensors + as_list(other_tensors)

        if len(target_outputs) > 0:
            invalid_out = [target for target in target_outputs if target not in self.output_layers]
            if len(invalid_out) != 0:
                raise ValueError("Invalid target outputs. outputs not in the graph:\n"
                                 "{outs}".format(outs="\n".join(map(str, invalid_out))))
            output_layers = target_outputs
            # input_layers = {dep for target_out in output_layers for dep in self.dependencies[target_out]}

        """
        if feed is None:
            if len(input_values) != len(input_layers):
                raise ValueError("number of input values ({n_input_values}) does "
                                 "not match number of graph input layers "
                                 "({n_input_layers})".format(n_input_values=len(input_values),
                                                             n_input_layers=len(input_layers)))

            feed = {}
            if use_defaults:
                default_feedable = [feedable for feedable in input_layers if feedable.value is not None]
            else:
                default_feedable = []

            if use_defaults:
                required_feedable = [feedable for feedable in input_layers if feedable.value is None]
                print("inputs")
                print(list(map(str, required_feedable)))
                print(input_values)
            else:
                required_feedable = input_layers

            if use_defaults:
                default_feed = {f_layer.placeholder: f_layer.value for f_layer in default_feedable}
                feed.update(default_feed)

            required_feed = {f_layer.placeholder: graph_input for f_layer, graph_input in
                             zip(required_feedable, input_values)}

            feed.update(required_feed)
            print(feed)
        else:
        """
        inputs_fed = set(feed.keys())

        missing_dependencies = self.missing_dependencies(inputs_fed, output_layers)
        missing_others = set()
        if use_defaults:
            new_missing_dep = {}
            default_feed = {}
            for out_layer, in_layers in missing_dependencies.items():
                for in_layer in in_layers:
                    if in_layer.value is not None:
                        default_feed[in_layer] = in_layer.value
                    else:
                        if out_layer not in new_missing_dep:
                            new_missing_dep[out_layer] = {in_layer}
                        else:
                            new_missing_dep[out_layer].add(in_layer)

            # feed other inputs (not connected to the graph necessarily)
            for in_layer in self.other_inputs:
                if in_layer.value is not None:
                    default_feed[in_layer] = in_layer.value
                else:
                    missing_others.add(in_layer)
            missing_dependencies = new_missing_dep
            feed.update(default_feed)

        if len(missing_others) > 0:
            dep_str = [str(missing) for missing in missing_others]
            raise ValueError("Could not run eval, missing other_inputs without default"
                             "values: \n {dep}".format("\n".join(dep_str)))

        if len(missing_dependencies) > 0:
            dep_str = [str(o) + "<---[{}]".format(",".join(map(str, i))) for o, i in missing_dependencies.items()]
            raise ValueError("Could not evaluate graph: \n Missing input dependencies:"
                             "\n {missing}".format(missing="\n".join(dep_str)))

        feed = {layer.tensor: data for layer, data in feed.items()}

        fetches = [out_layer.tensor for out_layer in output_layers] + other_tensors
        result = session.run(fetches=fetches,
                             feed_dict=feed,
                             options=options,
                             run_metadata=run_metadata)

        if len(output_layers) == 1 and len(other_tensors) == 0:
            result = result[0]

        return result

    def draw(self, path="layer_graph.pdf"):
        dg = AGraph(directed=True)

        for node in self.graph.nodes:
            dg.add_node(node.name)
        for node in self.graph.nodes:
            for other_node in self.graph.edges_out[node]:
                dg.add_edge(node.name, other_node.name)

        dg.layout(prog="dot")
        dg.draw(path=path)


class Model:
    """ Model.

    A `Model` is a container for TensorX graphs. It stores the endpoints (input-output) of a model
    and facilitates training, inference, and evaluation

    Args:

    """

    def __init__(self,
                 run_outputs,
                 run_inputs=None,
                 train_inputs=None,
                 train_outputs=None,
                 train_loss=None,
                 eval_inputs=None,
                 eval_outputs=None,
                 eval_score=None,
                 update_inputs=None,
                 update_ops=None,
                 name='Model'):
        self.name = name
        # run layers

        self.run_inputs = as_list(run_inputs)
        self.run_outputs = as_list(run_outputs)

        self.train_inputs = as_list(train_inputs)
        self.train_outputs = as_list(train_outputs)
        self.train_loss = as_list(train_loss)

        self.eval_inputs = as_list(eval_inputs)
        self.eval_outputs = as_list(eval_outputs)
        self.eval_score = as_list(eval_score)

        # this can be a set of params with default values
        self.update_inputs = as_list(update_inputs)
        self.update_ops = as_list(update_ops)

        self.run_graph: LayerGraph = LayerGraph(inputs=self.run_inputs,
                                                outputs=self.run_outputs)

        self.eval_graph: LayerGraph = LayerGraph(inputs=self.eval_inputs,
                                                 outputs=self.eval_outputs + self.eval_score)

        self.run_vars = {var.name for layer in self.run_graph.layers for var in layer.variables}
        self.train_vars = []
        self.eval_vars = {var.name for layer in self.eval_graph.layers for var in layer.variables}

        if len(self.update_ops) > 0:
            self.update_graph: LayerGraph = LayerGraph(inputs=self.update_inputs, outputs=self.update_ops)
        else:
            self.update_graph = None

        # model running init
        self.session = None

        # var inited = ([true|false], session)
        self._var_inited = (None, None)

        # properties for training
        self.optimizer = None
        self.optimizer_params = []
        self.joint_loss = None
        self.var_list = None
        self.train_step = None

        # op for model saving and restoring

        if self.has_vars():
            self.saver = Saver()
        self.init_var_op = None

        self.log_writer = None
        self.log_dir = None
        self.runtime_stats = None
        self.run_metadata = None

        self.run_options = None

        self.run_steps = 0
        self.train_steps = 0
        self.eval_steps = 0

        self.train_graph: LayerGraph = None
        self.train_called = False

    def has_vars(self):
        return (len(self.run_vars) != 0
                or len(self.train_vars) != 0
                or len(self.eval_vars) != 0)

    def set_log_dir(self, log_dir=None):
        self.log_dir = log_dir

        if self.log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), "log")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.log_dir) or not os.path.isdir(self.log_dir):
            raise ValueError("logdir {} does not exist or is not a directory".format(log_dir))

    def set_log_writer(self):
        # log dir changed, change writer
        if self.log_writer is None or not os.path.samefile(self.log_writer.get_logdir(), self.log_dir):
            self.log_writer = FileWriter(self.log_dir, self.session.graph)

    def close_logs(self):
        """ Closes log writers, etc
        """
        self.log_writer.close()

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

    def log_graph(self, logdir=None):
        """ outputs the graph meta file to be open in Tensorboard
        Args:
            logdir: path to directory where the graph is to be written

        """
        self.set_session()
        if self.log_dir is not None and logdir is None:
            logdir = self.log_dir
        self.set_log_dir(logdir)
        self.set_log_writer()
        self.log_writer.add_graph(self.session.graph)

    def save_model(self, logdir=None, model_name="model.ckpt", step=None, epoch=None, save_graph=False,
                   write_state=True):
        """ Saves all the variable states

        Note:
            if no session exists it creates a new default session

        Args:
            write_state: if true writes the checkpoint file with a list of all checkpoints
            save_graph: if true also exports the graph to model_Name.meta
            model_name: name for the model to be saved
            logdir: path to a ckpt file where the model is to be stored
            step: integer or tensor with the current step for the model checkpoint

        """

        if not (self.has_vars() or save_graph):
            raise ValueError("The model has no variables to save and save_graph was set to False: Nothing to save")

        if self.session is None:
            self.set_session()

        self.set_log_dir(logdir)
        model_path = os.path.join(self.log_dir, model_name)

        if save_graph:
            meta_path = "{model_path}.meta".format(model_path=model_path)
            export_meta_graph(meta_path)

        if self.has_vars():
            self.saver.save(self.session, model_path, step, write_meta_graph=False, write_state=write_state)

    def load_model(self, logdir=None, model_name="model.ckpt", global_step=None, load_graph=False):
        """ Loads the variables on the given path to the current graph, if
        global_step is provided loads that particular checkpoint (if it exists)
        otherwise tries to load the most recent checkpoint with the given name

        Note:
            if a current session does not exist, creates a new session.
            declares the current model as initialised

        Args:
            load_graph:
            global_step: step from which the model should be restored
            logdir: path to the directory where the model is to be saved
            model_name: the path where the model is to be restored
        """
        if self.session is None:
            self.set_session()

        self.set_log_dir(logdir)
        model_path = os.path.join(self.log_dir, model_name)

        if global_step is not None:
            if isinstance(global_step, Variable):
                step = self.session.run(global_step)
            model_path = "{path}-{i}".format(path=model_path, i=step)

        if load_graph:
            meta_path = "{model_path}.meta".format(model_path=model_path)
            self.saver = import_meta_graph(meta_path)

        if self.has_vars():
            self.saver.restore(self.session, model_path)
        # we don't need to init vars after loading a model
        self._set_vars_inited()

    def set_session(self, session=None, runtime_stats=False, run_options=None):
        """ Sets the session being used by :class:`Model` class.

        If no session is passed it sets the session as follows:
            1. sets the session to the default session if available
            2. creates a new session and uses it as the default session for the model class.

        Args:
            run_options:
            runtime_stats:
            session: a TensorFlow ``Session``.

        Returns:
            ``Session``: the current ``Session`` being used by the model class.

        """
        if session is not None and not isinstance(session, (Session, InteractiveSession)):
            raise TypeError("Expecting a TensorFlow Session object, got {} instead".format(type(session)))

        if session is None:
            session = _default_session()
        self.session = session

        if self.run_options is None:
            self.run_options = run_options

        if self.runtime_stats is None and runtime_stats:
            self.runtime_stats = runtime_stats
            self.run_metadata = RunMetadata()
            # setup default run options
            if self.run_options is None:
                self.run_options = RunOptions(trace_level=RunOptions.FULL_TRACE)

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

    def config_optimizer(self, optimizer, optimizer_params=None, gradient_op=None, global_gradient_op=False,
                         var_list=None):
        """ Configures the model for training

        # the idea is to add an op that can be applied to the gradients and output in the training method


        Note:
            I suspect we only need to process gradients directly (gradient clipping etc). If the use-case
            arises, we can modify this to accept a function that takes a list of (gradient,variable) tupples
            and returns a list of new  (gradientd,variable) tensors to be applied.

        Gradient OP Example:
            to apply a global gradient op like `tf.clip_by_global_norm`` would require the user to wrap this in a
            function that given a list of gradients produces a list of new gradient tensors:

            gradient_op: [grads] -> [grads]

        Args:
            global_gradient_op: if True applies gradient_op to the entire gradient list,
            if False calls gradient_op for each gradient in the list individually.
            var_list: list o variables modified by the optimizer, if None, the optimizer is applied to
            all variables marked as trainable.
            gradient_op : gradient op is to be applied to each gradient.
            optimizer_params: a :obj:`list` or single `Param` to be used with the optimizer, the feedable
            parameters should be fed by the same order in the train method

            optimizer: the tensorflow optimiser used to train the model
        """
        self.optimizer = optimizer
        self.optimizer_params = as_list(optimizer_params)
        self.var_list = var_list

        if len(self.train_loss) == 0:
            raise ValueError("Cannot add an optimizer: this model has no loss functions")
        elif len(self.train_loss) == 1:
            self.joint_loss = self.train_loss[0]
        else:
            self.joint_loss = Mean(*self.train_loss)

        def minimize(loss):
            if gradient_op is not None:
                grads_vars = self.optimizer.compute_gradients(loss, var_list=self.var_list)
                gradients, variables = zip(*grads_vars)

                if global_gradient_op:
                    new_gradients = gradient_op(gradients)
                else:
                    new_gradients = [None if g is None else gradient_op(g) for g in gradients]

                grads_vars = zip(new_gradients, variables)

                train_step = self.optimizer.apply_gradients(grads_vars)
            else:
                train_step = self.optimizer.minimize(loss, var_list=var_list)

            return train_step

        self.train_step = minimize(self.joint_loss.tensor)

        self.train_graph = LayerGraph(inputs=self.train_inputs,
                                      outputs=self.joint_loss,
                                      other_inputs=self.optimizer_params,
                                      other_tensors=self.train_step)

        self.train_vars = {var.name for layer in self.train_graph.layers for var in layer.variables}

    def run(self, feed, write_summaries=False):
        """ run the model (inference graph)
        """
        if self.session is None:
            self.set_session()

        if not self.vars_inited() and self.has_vars():
            self.init_vars()

        # make sure state is up to date before calling run
        if self.train_called:
            if self.update_graph is not None:
                g: LayerGraph = self.update_graph
                g.eval(use_defaults=True, session=self.session)
        self.train_called = False

        other_fetches = None
        if write_summaries:
            other_fetches = [summary.merge_all()]

        if self.runtime_stats:
            result = self.run_graph.eval(feed=feed,
                                         other_tensors=other_fetches,
                                         use_defaults=True,
                                         session=self.session,
                                         options=self.run_options,
                                         run_metadata=self.run_metadata)

            if self.log_dir is None:
                self.set_log_dir()
            self.set_log_writer()
            self.log_writer.add_run_metadata(self.run_metadata, tag="run step {}".format(self.run_steps + 1),
                                             global_step=self.run_steps + 1)
        else:
            result = self.run_graph.eval(feed=feed,
                                         other_tensors=other_fetches,
                                         use_defaults=True,
                                         session=self.session)

        if write_summaries:
            result, logs = result[0:-1], result[-1]
            self.log_writer.add_summary(logs, self.run_steps + 1)

        self.run_steps += 1

        # for convenience if we have a single output layer return the result, not a list of results
        if len(self.run_outputs) == 1:
            result = result[0]
        return result

    def train(self, feed_dict,
              write_summaries=False):
        """ Trains the model on the given data.

        Uses the configured optimiser and loss functions to train the update the model variables for n
        epochs.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Warning:
            You need to run :func:`config` before calling `train`.

        Args:
            feed_dict:
            write_summaries:
            output_loss:
            optimizer_params: values to be fed to the feedable ``Params`` specified in ``config_optimizer``
            model_input_data: a :obj:`list` of NumPy `ndarray` with the data to be fed to each model input
            loss_input_data: a :obj:`list` of NumPy `ndarray` with the data to be fed to `self.targets`.
        """
        self.train_called = True

        if self.session is None:
            self.set_session()

        if not self.vars_inited():
            self.init_vars()

        if self.train_graph is None:
            raise AttributeError("ModelRunner has no train graph, call configure_optimizer before train")

        other_fetches = []
        if write_summaries:
            other_fetches = as_list(summary.merge_all())

        # RUNTIME STATISTICS such as compute time, memory etc
        if self.runtime_stats:
            if self.log_dir is None:
                self.set_log_dir()
            self.set_log_writer()

            results = self.train_graph.eval(
                target_outputs=self.joint_loss,
                other_tensors=other_fetches,
                feed=feed_dict,
                use_defaults=True,
                session=self.session,
                options=self.run_options,
                run_metadata=self.run_metadata)

            self.log_writer.add_run_metadata(self.run_metadata,
                                             tag="train step {}".format(self.train_steps + 1),
                                             global_step=self.train_steps + 1)

        else:
            results = self.train_graph.eval(
                target_outputs=self.joint_loss,
                other_tensors=other_fetches,
                feed=feed_dict,
                use_defaults=True,
                session=self.session)

        result, other = results[0], results[1:]

        if write_summaries and len(other_fetches) > 0:
            logs = other[-1]
            self.log_writer.add_summary(logs, self.train_steps + 1)

        self.train_steps += 1

        return result

    def eval(self, feed, write_summaries=False):
        """ Evaluates the model on the given data.

        If multiple loss functions are provided, it performs joint training by summing the loss functions.

        Args:
            feed (dict): dictionary with eval graph dependencies layer: value
            write_summaries:
        """

        if self.session is None:
            self.set_session()

        if not self.vars_inited() and self.has_vars():
            self.init_vars()

            # make sure state is up to date before calling run
        if self.train_called:
            if self.update_graph is not None:
                g: LayerGraph = self.update_graph
                g.eval(use_defaults=True, session=self.session)
        self.train_called = False

        other_fetches = None
        if write_summaries:
            other_fetches = as_list(summary.merge_all())

        if self.runtime_stats:
            result = self.eval_graph.eval(
                target_outputs=self.eval_score,
                other_tensors=other_fetches,
                use_defaults=True,
                feed=feed,
                session=self.session,
                options=self.run_options,
                run_metadata=self.run_metadata)

            if self.log_dir is None:
                self.set_log_dir()
            self.set_log_writer()
            self.log_writer.add_run_metadata(self.run_metadata, tag="eval step {}".format(self.eval_steps + 1),
                                             global_step=self.eval_steps + 1)
        else:
            result = self.eval_graph.eval(
                target_outputs=self.eval_score,
                feed=feed,
                other_tensors=other_fetches,
                use_defaults=True,
                session=self.session)

        if write_summaries and len(other_fetches) > 0:
            result, logs = result[0:-1], result[-1]
            self.log_writer.add_summary(logs, self.eval_steps + 1)

        self.eval_steps += 1

        return result


class EvalStepDecayParam(DynamicParam):
    """
    Args:
        value: initial value for the dynamic param
        decay_threshold: float value representing the difference between evaluations necessary for the update to occur
        decay_rate: rate through which the param value is reduced `(value = value * decay_rate)`
        decay_threshold: point beyond witch the param value is not reduced `max(value * decay_rate, decay_threshold)`
        less_is_better: if True, evaluation is considered to improve if it decreases, else it is considered to improve
        if it increases

    Attributes:
        eval_history: a list with the evaluation values passed through the update function
        improvement_threshold: float value representing the difference between evaluations necessary for the update to occur
        decay_rate: rate through which the param value is reduced `(value = value * decay_rate)`
        decay_threshold: point beyond witch the param value is not reduced `max(value * decay_rate, decay_threshold)`
    """

    def __init__(self, value,
                 improvement_threshold=1.0,
                 less_is_better=True,
                 decay_rate=1.0,
                 decay_threshold=1e-6,
                 dtype=dtypes.float32,
                 name="eval_step_decay_param"):
        self.improvement_threshold = improvement_threshold
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.less_is_better = less_is_better
        self.eval_history = []

        def update_fn(evaluation):
            self.eval_history.append(evaluation)
            value = self.value
            if len(self.eval_history) > 1:
                if self.eval_improvement() <= self.improvement_threshold:
                    value = max(value * self.decay_rate, self.decay_threshold)
            return value

        super().__init__(value=value, dtype=dtype, update_fn=update_fn, name=name)

    def eval_improvement(self):
        if len(self.eval_history) > 1:
            improvement = self.eval_history[-2] - self.eval_history[-1]
            if not self.less_is_better:
                improvement = -1 * improvement
            return improvement
        else:
            return 0

    def update(self, evaluation):
        """ update.

        Updates the parameter value. It only makes changes to the parameter after then second update.
        The decay decays is only applied if the current evaluation did not improve more than the `eval_threshold`.

        Args:
            evaluation: a float with the current evaluation value
        """
        super().update(evaluation)


class StepDecay(DynamicParam):
    """
    """

    def __init__(self, value,
                 decay_after=0,
                 decay_rate=1.0,
                 decay_threshold=1e-6,
                 dtype=dtypes.float32,
                 name="step_decay_param"):

        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.decay_after = decay_after
        self.current_step = 0

        def update_fn():
            self.current_step += 1
            if self.current_step <= decay_after:
                new_value = self.value
            else:
                new_value = max(self.decay_threshold, self.value * decay_rate)

            return new_value

        super().__init__(value=value, dtype=dtype, update_fn=update_fn, name=name)


__all__ = ["Model",
           "LayerGraph",
           "EvalStepDecayParam"]
