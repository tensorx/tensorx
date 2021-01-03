import tensorflow as tf
import numpy as np
import logging
from collections import Counter

logging.captureWarnings(True)  # captures into py.warnings
logger = logging.getLogger('tensorx')
logger.setLevel(logging.DEBUG)
import re


class Graph:
    """ Graph

    Simple append-only graph data structure. It keeps track of nodes, directed edges, and endpoint nodes.

    Note:
        A node without edges counts as both an input and output node of the graph

    Attributes:
        nodes(set): set of node objects (object need tobe hashable)
        edges_in(dict): a dictionary that maps nodes in the keys to a list of notes with edges coming into that node
        edges_out(dict): a dictionary that maps nodes in the keys to a list of notes with edges coming from that node

        in_nodes(dict): key-only dictionary (ordered set) with input nodes (nodes without input edges).
        out_nodes(dict): key-only dictionary (ordered set) with output nodes of the graph (nodes without output edges)
    """

    def __init__(self):
        self.nodes = set()
        self.in_nodes = dict()
        self.out_nodes = dict()
        self.edges_in = dict()
        self.edges_out = dict()

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges_in[node] = []
            self.edges_out[node] = []

            # dict without values is a special case of ordered set
            self.in_nodes[node] = None
            self.out_nodes[node] = None

    def add_edge(self, node1, node2):
        """ Adds a new edge to the graph

        also removes nodes from input roots or outputs to reflect the current edge if necessary.

        Args:
            node1 (`Node`): starting node
            node2 (`Node`): ending node
        """
        self.add_node(node1)
        self.add_node(node2)
        self.edges_out[node1].append(node2)
        self.edges_in[node2].append(node1)

        # update endpoints
        if node1 in self.out_nodes:
            del self.out_nodes[node1]
        if node2 in self.in_nodes:
            del self.in_nodes[node2]

    def dependency_iter(self):
        """ returns a dictionary with a map from nodes to dependency priorities
            with lower values having higher priority. Keys are ordered by priority from
            lower to higher and number of dependencies from lower to higher

            Notes:
               Transversing a graph by priority guarantees that when we visit a node
               all it's dependencies have already been visited, additionally, ordering by
               number of dependencies guarantees that we can maintain a minimum result
               cache when transversing the graph.

           Returns:
               nodes (`dict`): dictionary from nodes to (priorities,number of dependencies)
        """
        priority = dict()
        visited = set()
        nodes = list(self.in_nodes)

        while nodes:
            current = nodes.pop(0)
            visited.add(current)
            delayed = False

            if not self.edges_in[current]:
                priority[current] = (0, len(self.edges_out[current]))
            else:
                # delay node if not all dependencies are ready
                if any([dep not in priority for dep in self.edges_in[current]]):
                    nodes.append(current)
                    delayed = True
                else:
                    priorities = [priority[dep][0] for dep in self.edges_in[current]]
                    priority[current] = (max(priorities) + 1, len(self.edges_out[current]))

            if not delayed:
                next_nodes = self.edges_out[current]
                for next_node in next_nodes:
                    if next_node not in visited:
                        nodes.insert(0, next_node)

        sorted_priority = dict(sorted(priority.items(), key=lambda kv: kv[1], reverse=False))

        return sorted_priority

    # TODO this doesn't take Tensors, only layers
    @staticmethod
    def build(inputs, outputs, add_missing_inputs=False):
        """ build_graph

        !!! note
            use `add_missing_inputs` if you have graph inputs but might have other dependencies that might not have
            been created explicitly. Example: in an RNN layer, if a previous state is not passed explicitly, a default
            one is created by the layer and stored in input layers. You might be aware of this input node to a graph but
            not want to pass it explicitly to inputs.

        Args:
            inputs: input terminal layers where the graph must stop
            outputs: output layers from which we start to populate the graph
            add_missing_inputs: if True and `inputs` are provided, input nodes found that are not in given inputs
                will be added to the graph. If False ValueError is raised with a list of inputs not specified
                (missing dependencies).

        Returns:
            graph (`Graph`): a graph from the outputs to the given input, or to every input found if these are not
                specified.
        """
        graph = Graph()
        inputs = dict.fromkeys(as_list(inputs))
        graph.outputs = as_list(outputs)
        outputs = dict.fromkeys(as_list(outputs))

        # add terminals to the graph
        # for layer in input_layers:
        #    graph.add_node(layer)
        for layer in outputs:
            graph.add_node(layer)

        dependencies = dict()
        missing_dependencies = dict()

        def add_dep(out, dep, target: dict):
            if out not in target:
                target[out] = set()
            target[out].add(dep)

        # arg order in a path to the output
        arg_ord = {out: (0,) for out in outputs}
        visited = set()
        node_queue = list(zip(outputs, outputs))

        while node_queue:
            current_node, target_output = node_queue.pop(0)
            if current_node not in visited:
                next_nodes = current_node.inputs
                if not next_nodes:
                    add_dep(target_output, current_node, dependencies)
                    if len(inputs) > 0 and current_node not in inputs and not add_missing_inputs:
                        add_dep(target_output, current_node, missing_dependencies)
                else:
                    if current_node in inputs:
                        add_dep(target_output, current_node, dependencies)
                    else:
                        for i, input_node in enumerate(next_nodes):
                            graph.add_edge(input_node, current_node)
                            node_queue.append((input_node, target_output))

                            # record arg order
                            if input_node not in arg_ord:
                                arg_ord[input_node] = arg_ord[current_node] + (i + 1,)

                visited.add(current_node)

        if any(missing_dependencies) and not add_missing_inputs:
            failed_str = []
            for output_layer in missing_dependencies:
                missing_str = "\n\t\t".join(map(str, missing_dependencies[output_layer]))
                failed_str.append(f"\t{str(output_layer)}: \n\t\t{missing_str}")
            failed_str = "\n".join(failed_str)

            raise ValueError(f"output layers missing inputs: \n {failed_str}")

        if inputs:
            missing_from_graph = list(filter(lambda x: x not in graph.in_nodes, inputs))

            if missing_from_graph:
                input_str = "\n\t ".join(map(str, missing_from_graph))
                output_str = "\n\t ".join(map(str, outputs))
                raise ValueError(f"no path between the output layers:\n\t {output_str} \n and input layers: \n\t"
                                 f"{input_str}")

        inputs.update(graph.in_nodes)
        outputs.update(graph.out_nodes)

        # if no ordered input is given
        # re-order by argument ordering
        if not inputs:
            # sort according to argument ordering from the outputs
            sorted_inputs = sorted(inputs, key=lambda in_layer: arg_ord[in_layer], reverse=False)
            inputs = dict.fromkeys(sorted_inputs)

        graph.in_nodes = inputs
        graph.out_nodes = outputs

        return graph

    def as_function(self, ord_inputs=None, ord_outputs=None, name="compiled_graph", compile=True):
        """ compiles the graph into a tensorflow callable compiled graph

        Converts the current graph into a function with a series of `layer.compute(*tensors)` calls
        and uses `tf.function` to compile this function to a Tensorflow static graph if compile is `True`.
        The resulting function is a closure with access to layer objects, to TensorFlow should be able to
        trace the computations for each layer `compute` call.


        Another way to feed inputs to a graph is to use input layers and change the value, if the graphs are created
        without inputs, but the terminal input nodes are Dynamic Inputs, the execution of those layers is a read
        on their placeholder value, which you can change that value before calling the graph and the output will be
        correct.

        ```python
        input_layer.value = in0
        input_Layer.value = in1
        outputs = graph()
        ```

        this adds a bit of a overhead since we have to write to the variable

        !!! bug "Dev Note"
            * makes use of `dependency_iter` to create the computation calls such that when we call compute all the
                inputs needed as dependencies are already available.


        Args:
            ord_inputs (`List[Node]`): list of input that determines the order of resulting function arguments
            ord_outputs (`List[Node`]): list of outputs used to determine the return order
            name (`str`): function name, must be a valid python function name
            compile (`bool`): if True, returns a tensorflow graph else returns a python function

        Returns:
            function (`Callable`): an optimized TensorFlow static graph as a callable function or a python function

        """

        clean = lambda name_str: re.sub(r"\W|^(?=\d)", "_", name_str)
        name = clean(name)

        graph = self

        if not graph.out_nodes:
            raise ValueError("can't compile an empty graph")
        ord_inputs = as_list(ord_inputs)
        ord_outputs = as_list(ord_outputs)
        ord_nodes = list(self.dependency_iter())

        input_set: set = set(graph.in_nodes)
        if ord_inputs and not input_set.issuperset(ord_inputs):
            raise ValueError("all feedable_inputs must be part of the graph inputs")
        output_set: set = set(graph.out_nodes)
        if ord_outputs and len(output_set.difference(ord_outputs)) > 0:
            raise ValueError("all outputs must be part of the graph outputs")

        # if no input order is specified use the graph endpoint order
        outputs = dict.fromkeys(ord_outputs) if ord_outputs else graph.out_nodes

        # if we don't provide inputs it will just treat them as callables
        inputs = dict.fromkeys(ord_inputs) if ord_inputs else []  # graph.in_nodes

        # check if they are all dynamic inputs
        # in py3.7 the dict is an ordered set if we convert it back to a list
        node_index = list(range(len(graph.nodes)))

        feedable_inputs = list(inputs)
        node_map = {}
        for in_layer in feedable_inputs:
            layer_i = node_index.pop(0)
            in_name = in_layer.name.replace('/', '__')
            layer_name = f"{in_name}_{layer_i}"
            node_map[in_layer] = layer_name

        args_str = ", ".join(node_map.values())
        def_str = f"def {name}({args_str}):\n"
        other_str = []

        # all other inputs that are not feedable
        other_inputs = list(input_set.difference(feedable_inputs))
        node_map.update({in_layer: f"{in_layer.name}_{node_index.pop(0)}" for in_layer in other_inputs})

        # requires outer access to layers var
        for x in other_inputs:
            other_str.append(f"\t{node_map[x]} = layers[\"{node_map[x]}\"].compute()")

        other_str = "\n".join(other_str) + "\n" if other_str else ""

        # remove inputs
        # node_map contains input_nodes at this point
        for _ in range(len(node_map)):
            ord_nodes.pop(0)

        compute_str = []
        for current_node in ord_nodes:
            node_name = current_node.name.replace('/', '__')
            node_map[current_node] = f"{node_name}_{node_index.pop(0)}"
            node_name = node_map[current_node]
            # when layers have the same layer repeated as input, this causes problems
            # it's better to use the same input_layers as declared in the graph
            # dict from keys creates an ordered set which is not what we want
            # next_nodes = dict.fromkeys(graph.edges_in[current_node])
            next_nodes = graph.edges_in[current_node]
            in_args = ", ".join([node_map[node] for node in next_nodes])
            compute_str.append(f"\t{node_name} = layers[\"{node_name}\"].compute({in_args})")

        compute_str = "\n".join(compute_str)

        return_str = "\n\treturn {output_str}\n".format(output_str=", ".join([node_map[out] for out in outputs]))

        full_fn_str = def_str + other_str + compute_str + return_str
        logger.log(logging.DEBUG, f"converted function:\n {'-' * 10}\n\n {full_fn_str} \n{'-' * 10}")
        # layer map (for the closure above)
        # we feed the locals so that layers gets available in the above function
        layers = {v: k for k, v in node_map.items()}
        exec(full_fn_str, locals())
        fn = eval(name)
        fn.__doc__ = f"""{name}\n```python\n{full_fn_str}\n```"""

        if compile:
            fn = tf.function(fn)

        return fn

    def as_function_v2(self,
                       ord_inputs=None,
                       ord_outputs=None,
                       fn_name="compiled_graph",
                       stateful_inputs=False,
                       compile=True):
        # this could be static
        graph = self
        fn_name = re.sub(r"\W|^(?=\d)", "_", fn_name)

        if not graph.out_nodes:
            raise ValueError("can't compile an empty graph")
        ord_inputs = as_list(ord_inputs)
        ord_outputs = as_list(ord_outputs)
        ord_nodes = list(self.dependency_iter())

        input_set: set = set(graph.in_nodes)
        if ord_inputs and not input_set.issuperset(ord_inputs):
            raise ValueError("all feedable_inputs must be part of the graph inputs")
        output_set: set = set(graph.out_nodes)
        if ord_outputs and len(output_set.difference(ord_outputs)) > 0:
            raise ValueError("all outputs must be part of the graph outputs")

        # if no input order is specified use the graph endpoint order
        outputs = dict.fromkeys(ord_outputs) if ord_outputs else graph.out_nodes

        # if we don't provide inputs it will just treat them as callables
        inputs = dict.fromkeys(ord_inputs) if ord_inputs else []  # graph.in_nodes

        # check if they are all dynamic inputs
        # in py3.7 the dict is an ordered set if we convert it back to a list
        node_index = list(range(len(graph.nodes)))

        feedable_inputs = list(inputs)
        node_map = {}
        for in_layer in feedable_inputs:
            layer_i = node_index.pop(0)
            in_name = in_layer.name.replace('/', '__')
            layer_name = f"{in_name}_{layer_i}"
            node_map[in_layer] = layer_name

        # TODO check if these can be inputs that are not Input layers?
        # if using stateful inputs, we need to be able to pass no args
        fn_args = []
        update_inputs = []
        for layer, layer_name in node_map.items():
            if stateful_inputs and feedable_inputs:
                fn_args.append(f"{layer_name}=None")
                if not layer.constant:
                    update_inputs.append(f"\tif {layer_name}:\n\t\tlayers[\"{layer_name}\"].value = {layer_name}"
                                         f"\n\telse:\n"
                                         f"\t\t{layer_name} = layers[\"{layer_name}\"].compute()")
                else:
                    update_inputs.append(f"\tif not {layer_name}:\n"
                                         f"\t\t{layer_name} = layers[\"{layer_name}\"].compute()")

            else:
                fn_args.append(f"{layer_name}")

        args_str = ", ".join(fn_args)
        # args_str = ", ".join(node_map.values())
        def_str = f"def {fn_name}({args_str}):\n"

        update_str = "\n".join(update_inputs) + "\n"

        other_str = []

        # all other inputs that are not feedable
        other_inputs = list(input_set.difference(feedable_inputs))
        node_map.update({in_layer: f"{in_layer.name}_{node_index.pop(0)}" for in_layer in other_inputs})

        # requires outer access to layers var
        for x in other_inputs:
            other_str.append(f"\t{node_map[x]} = layers[\"{node_map[x]}\"].compute()")

        other_str = "\n".join(other_str) + "\n" if other_str else ""

        # remove inputs
        # node_map contains input_nodes at this point
        for _ in range(len(node_map)):
            ord_nodes.pop(0)

        compute_str = []
        for current_node in ord_nodes:
            name = current_node.name.replace('/', '__')
            node_map[current_node] = f"{name}_{node_index.pop(0)}"
            name = node_map[current_node]
            # when layers have the same layer repeated as input, this causes problems
            # it's better to use the same input_layers as declared in the graph
            # dict from keys creates an ordered set which is not what we want
            # next_nodes = dict.fromkeys(graph.edges_in[current_node])
            next_nodes = graph.edges_in[current_node]
            in_args = ", ".join([node_map[node] for node in next_nodes])
            compute_str.append(f"\t{name} = layers[\"{name}\"].compute({in_args})")

        compute_str = "\n".join(compute_str)

        return_str = "\n\treturn {output_str}\n".format(output_str=", ".join([node_map[out] for out in outputs]))

        full_fn_str = def_str + update_str + other_str + compute_str + return_str
        logger.log(logging.DEBUG, f"converted function:\n {'-' * 10}\n\n {full_fn_str} \n{'-' * 10}")

        # layer map (for the closure above)
        # we feed the locals so that layers gets available in the above function
        layers = {v: k for k, v in node_map.items()}
        print("\n", full_fn_str)
        exec(full_fn_str, locals())
        fn = eval(fn_name)
        fn.__doc__ = f"""{fn_name}\n```python\n{full_fn_str}\n```"""

        if compile:
            fn = tf.function(fn)

        return fn

    def draw(self, path):
        try:
            from pygraphviz import AGraph

            def vizstyle(layer_obj):
                # HTML for record nodes https://graphviz.org/doc/info/shapes.html#top

                dtype = layer_obj.dtype.name if layer_obj.dtype is not None else None

                label = f"<<TABLE BORDER=\"0\"" \
                        f"        CELLPADDING=\"2\"" \
                        f"        CELLSPACING=\"0\">" \
                        f"<TR><TD BGCOLOR=\"BLACK\"" \
                        f"        BORDER=\"1\"" \
                        f"        COLOR=\"BLACK\"" \
                        f"        VALIGN=\"BOTTOM\">" \
                        f"<FONT COLOR=\"WHITE\"><B>{type(layer_obj).__name__}</B></FONT>" \
                        f"</TD><TD BORDER=\"1\">{layer_obj.name}</TD></TR>" \
                        f"<TR>" \
                        f"<TD BORDER=\"1\"" \
                        f"    BGCOLOR=\"#aec4c7\"" \
                        f"    COLOR=\"BLACK\"" \
                        f"    ALIGN=\"RIGHT\">" \
                        f"units" \
                        f"</TD><TD BORDER=\"1\"" \
                        f"         COLOR=\"BLACK\"" \
                        f"         ALIGN=\"LEFT\">" \
                        f"{layer_obj.n_units}</TD></TR>" \
                        f"<TR>" \
                        f"<TD BORDER=\"1\" " \
                        f"    BGCOLOR=\"#aec4c7\"" \
                        f"    ALIGN=\"RIGHT\">dtype</TD>" \
                        f"<TD BORDER=\"1\"" \
                        f"    ALIGN=\"LEFT\">{dtype}</TD></TR>" \
                        f"</TABLE>>"
                return label

            viz_graph = AGraph(directed=True)

            for node in self.dependency_iter():
                if node not in viz_graph:
                    viz_graph.add_node(node.name, shape="none", margin=0,
                                       label=vizstyle(node))  # , label=vizstyle(node))
                for other_node in self.edges_out[node]:
                    viz_graph.add_edge(node.name, other_node.name)

            dependencies = dependency_graph(self.nodes)
            state_deps = [node for node in dependencies.nodes if not dependencies.edges_in[node] and
                          len(dependencies.edges_out[node]) > 1]
            for node in state_deps:
                if hasattr(node, "name"):
                    name = node.name
                else:
                    # might be a tf.Variable
                    try:
                        name = node.deref().name
                    except AttributeError as e:
                        name = str(type(node).__name__)

                viz_graph.add_node(name, color="red", shape="box")
                layers = dependencies.edges_out[node]
                for layer in layers:
                    viz_graph.add_edge(name, layer.name, color="red")

            viz_graph.layout(prog="dot")
            viz_graph.draw(path=path)

        except ImportError:
            raise ImportError("Could't find required pygraphviz module")

    def compute(self, *input_values):
        """ computes the graph output values based on the given input values

        Args:
            *input_values: input values with the same order as the graph inputs, or a dictionary mapping values to
            input layers.

        Returns:
            a tuple with the values for the correspondent graph outputs
        """
        if len(input_values) == 1 and isinstance(input_values[0], dict):
            input_dict = input_values[0]
            missing = list(filter(lambda x: x not in self.in_nodes, input_dict.keys()))

            if missing:
                missing_str = '\n\t'.join([f"{str(x)}" for x in missing])
                raise ValueError(f"inputs not found in graphs:\n"
                                 f"\t{missing_str}")

            ord_inputs = dict.fromkeys(list(self.in_nodes)[:len(input_dict)])
            input_values = [as_tensor(input_dict[input_layer]) for input_layer in ord_inputs]

        elif len(input_values) > len(self.in_nodes):
            raise ValueError(f"too many inputs:\n"
                             f"\tgraph expects {len(self.in_nodes)} inputs\n"
                             f"\tinputs passed {len(input_values)}")
        else:
            input_values = [as_tensor(input_value) for input_value in input_values]

        ord_inputs = dict.fromkeys(list(self.in_nodes)[:len(input_values)])
        input_dict = dict(zip(ord_inputs.keys(), input_values))
        other_inputs = set(self.in_nodes).difference(ord_inputs)

        node_iter = self.dependency_iter()

        result_cache = dict()
        visited = set()

        for node in node_iter:
            if node in input_dict:
                result_cache[node] = input_dict[node]
            elif node in other_inputs:
                result_cache[node] = node.compute()
            else:
                visited.add(node)

                # get input_node result, clean cache when no more dependencies on the same input
                def get_args(node):
                    args = []
                    ins = self.edges_in[node]
                    for in_node in ins:
                        res = result_cache[in_node]
                        priority, num_deps = node_iter[node]
                        node_iter[node] = (priority, num_deps - 1)
                        if num_deps - 1 == 0:
                            del result_cache[in_node]
                        args.append(res)
                    return args

                args = get_args(node)
                result_cache[node] = node.compute(*args)
                # result_cache[node] = node(*args)

        return tuple(map(lambda x: result_cache[x], self.out_nodes))

    def __call__(self, *input_values):
        return self.compute(*input_values)

    @classmethod
    def eval(cls, *layers):
        graph = Graph.build(inputs=None, outputs=layers)
        return graph()


def dependency_graph(layers):
    g = Graph()
    for layer in layers:
        state = layer.layer_state
        layer_vars = layer.variables

        # g.add_edge(state, layer)
        for var in layer_vars:
            ref = var.ref()
            g.add_edge(ref, layer)
    return g


def as_numerical_shape(shape: tf.TensorShape):
    return [-1 if dim is None else dim for dim in shape]


def as_tensor(x, dtype=None):
    """ Converts to tensor and casts to a given type if possible

    Args:
        x: an input ``Tensor``.
        dtype: the type we which to cast the input tensor into

    Returns:
        ``Tensor``: a tensor with the given dtype
    """
    if dtype is not None:
        dtype = tf.dtypes.as_dtype(dtype)
    if not isinstance(x, tf.SparseTensor):
        x = tf.convert_to_tensor(x)

    if dtype is not None:
        if x.dtype != dtype:
            x = tf.cast(x, dtype)
    return x


def as_list(items):
    """ Returns a list from one or multiple elements.

    if one element is passed, returns a list with one element,
    if a list,a tuple or a dictionary of elements is passed,
    returns a list with the elements or the keys if the input is a dict

    Note: we exclude SparseTensorValue because it is a named tuple
    and we want to feed the whole object as a single data sample if needed

    Args:
        items: one item, a tuple of elements or a list of elements

    Returns:
        a :obj:`list` with the elements in items
    """
    if items is None:
        items = []
    elif isinstance(items, (list, tuple, dict)) and not isinstance(items, (
            tf.compat.v1.SparseTensorValue, tf.SparseTensor)):
        items = list(items)
    else:
        items = [items]
    return items


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        x_name = "(eager Tensor)"
        try:
            x_name = x.name
        except AttributeError:
            pass
    return cast_x


def fix_reshape_dimensions(original_shape, target_shape):
    """ Find and replace a missing dimension in a target shape.

    Args:
      original_shape (`List[int]`): shape of tensor being reshaped
      target_shape (`List`): desired shape with at most a single -1
        which indicates a dimension that should be derived from the input shape.

    Returns:
        new_shape (`List`): the new shape with a -1 replaced with its computed value.

    Raises:
      `ValueError`: if the total tensor size of the new_shape is different than
      the original_shape, or more than one unknown (-1) dimension are specified.
    """
    target_shape = list(target_shape)

    target_n = 1
    target_unknown = None
    for i, dim in enumerate(target_shape):
        if dim < 0:
            if target_unknown is None:
                target_unknown = i
            else:
                raise ValueError('Can only specify one unknown dimension.')
        else:
            target_n *= dim

    msg = ('total size of new tensor must be unchanged, '
           'input_shape = {}, output_shape = {}'
           .format(original_shape, target_shape))

    # the target should match original known
    original_known = [dim if dim else 1 for dim in original_shape]
    original_n = np.prod(original_known, dtype=int, keepdims=False)
    if target_unknown is not None:
        if target_n == 0 or original_n % target_n != 0:
            raise ValueError(msg)
        elif None in original_shape and len(original_shape) != len(target_shape) and original_n > target_n:
            target_shape[target_unknown] = None
        else:
            target_shape[target_unknown] = original_n // target_n
    elif original_n != target_n:
        raise ValueError(msg)
    return target_shape
