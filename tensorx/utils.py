import tensorflow as tf


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

    @staticmethod
    def merge(graph1, graph2):
        new_graph = Graph()

        new_graph.nodes.update(graph1.nodes)
        new_graph.in_nodes.update(graph1.in_nodes)
        new_graph.out_nodes.update(graph1.out_nodes)
        new_graph.edges_in.update(graph1.edges_in)
        new_graph.edges_out.update(graph1.edges_out)

        new_graph.nodes.update(graph2.nodes)
        new_graph.in_nodes.update(graph2.in_nodes)
        new_graph.out_nodes.update(graph2.out_nodes)
        new_graph.edges_in.update(graph2.edges_in)
        new_graph.edges_out.update(graph2.edges_out)

        return new_graph

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
        lower to higher.

        Transversing a graph by priority guarantees that when we visit a node
        all it's dependencies have already been visited.

        Returns:
            dictionary from nodes to priorities
        """
        priority = dict()
        max_priority = 0
        visited = set()
        nodes = list(self.out_nodes)
        while nodes:
            current = nodes.pop(0)
            if current not in visited:
                if current not in priority:
                    priority[current] = 0
                next_nodes = self.edges_in[current]
                for next_node in next_nodes:
                    p = priority[current] + 1
                    priority[next_node] = p
                    max_priority = max(p, max_priority)
                    nodes.append(next_node)
                visited.add(current)

        rev = list(range(max_priority + 1))
        rev.reverse()

        sorted_priority = {k: rev[v] for k, v in sorted(priority.items(), key=lambda kv: kv[1], reverse=True)}

        return sorted_priority

    @staticmethod
    def build(inputs, outputs, missing_inputs=False):
        """ build_graph

        Args:
            missing_inputs: if True and input_layers is not empty, missing input dependencies will be added to the graph
                else, having missing inputs will raise a ValueError exception listing the missing dependencies.
            inputs: input terminal layers where the graph must stop
            outputs: output layers from which we start to populate the graph

        Returns:
            a graph from the output layers to the given input layers
        """
        graph = Graph()
        inputs = dict.fromkeys(as_list(inputs))
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

        visited = set()
        node_queue = list(zip(outputs, outputs))

        while node_queue:
            current_node, target_output = node_queue.pop(0)
            if current_node not in visited:
                next_nodes = current_node.input_layers
                if not next_nodes:
                    add_dep(target_output, current_node, dependencies)
                    if len(inputs) > 0 and current_node not in inputs and not missing_inputs:
                        add_dep(target_output, current_node, missing_dependencies)
                else:
                    if current_node in inputs:
                        add_dep(target_output, current_node, dependencies)
                    else:
                        for input_node in next_nodes:
                            graph.add_edge(input_node, current_node)
                            node_queue.append((input_node, target_output))

                visited.add(current_node)

        if any(missing_dependencies) and not missing_inputs:
            failed_str = []
            for output_layer in missing_dependencies:
                missing_str = "\n\t\t".join(map(str, missing_dependencies[output_layer]))
                failed_str.append(f"\t{str(output_layer)}: \n\t\t{missing_str}")
            failed_str = "\n".join(failed_str)

            raise ValueError(f"output layers missing inputs: \n {failed_str}")

        if inputs:
            missing_from_graph = list(filter(lambda x: x not in graph.in_nodes, inputs))

            if missing_from_graph:
                raise ValueError("no path between the output layers and input layers: \n\t"
                                 "{}".format("\n\t ".join(map(str, missing_from_graph))))

        # re-order inputs and outputs according to the specification
        inputs.update(graph.in_nodes)
        outputs.update(graph.out_nodes)

        graph.in_nodes = inputs
        graph.out_nodes = outputs

        return graph

    def compile(self, ord_inputs=None, ord_outputs=None):
        """ compiles the graph into a tensorflow callable compiled graph

        the idea is to use exec to create a function and then call tf.function
        on the created function. this is to avoid using loops and lists to
        run through the Graph instance and call compute.

        function parameters:
            converts all the non-constant Input Layer nodes into graph arguments

        run through each layer and queue up nodes starting on the input
        write down a function as a series of compute calls with inputs from the previous
        layer outputs

        Args:
            ord_inputs: list of input that determines the order of resulting function arguments
            ord_outputs: list of outputs used to determine the return order
            feedable input in the compiled graph

        Returns:
            a callable tensorflow graph

        """
        # NOTE another way to feed inputs is to use input layers normally like
        #   input_layer.value = in0
        #   input_Layer.value = in1
        #   that way the input slots are up to date
        #   I guess this adds a bit of a overhead since we have to write to the variable

        graph = self

        if not graph.out_nodes:
            raise ValueError("can't compile an empty graph")
        ord_inputs = as_list(ord_inputs)
        ord_outputs = as_list(ord_outputs)

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
        node_map = {in_layer: f"{in_layer.name.replace('/', '__')}_{node_index.pop(0)}" for in_layer in feedable_inputs}
        args_str = ", ".join(node_map.values())
        def_str = f"def compiled_graph({args_str}):\n"
        other_str = []

        # all other inputs that are not feedable
        other_inputs = list(input_set.difference(feedable_inputs))
        node_map.update({in_layer: f"other_inputs_{node_index.pop(0)}" for in_layer in other_inputs})

        # requires outer access to layers var
        for x in other_inputs:
            other_str.append(f"\t{node_map[x]} = layers[\"{node_map[x]}\"].compute()")

        other_str = "\n".join(other_str)

        # create return and outputs
        for node in outputs:
            name = f"out_{node_index.pop()}"
            node_map[node] = name

        return_str = "\n\treturn {output_str}\n".format(output_str=", ".join([node_map[out] for out in outputs]))

        # for each layer not in inputs
        visited = set(graph.in_nodes)
        to_visit = set()
        node_queue = list(outputs)

        compute_str = []
        while node_queue:
            current_node = node_queue.pop(0)
            if current_node not in visited:
                next_nodes = dict.fromkeys(graph.edges_in[current_node])
                for node in next_nodes:
                    if node not in visited and node not in to_visit:
                        name = node.name.replace('/', '__')
                        node_map[node] = f"{name}_{node_index.pop()}"
                        node_queue.append(node)
                        to_visit.add(node)

                name = node_map[current_node]
                in_args = ", ".join([node_map[node] for node in next_nodes])
                compute_str.append(f"\t{name} = layers[\"{name}\"].compute({in_args})")
                visited.add(current_node)

        compute_str.reverse()
        compute_str = "\n".join(compute_str)

        full_fn_str = def_str + other_str + "\n" + compute_str + return_str
        # print(full_fn_str)
        # layer map (for the closure above)
        # we feed the locals so that layers gets available in the above function
        layers = {v: k for k, v in node_map.items()}
        exec(full_fn_str, locals())
        fn = eval("compiled_graph")
        out = tf.function(fn)

        return out

    # def compile_recursive(self, ord_inputs):
    #     ord_inputs = dict.fromkeys(as_list(ord_inputs))
    #     other_inputs = set(self.in_nodes).difference(ord_inputs)
    #     in_map = {node: i for i, node in enumerate(ord_inputs)}
    #
    #     @tf.function
    #     def compiled_graph(*inputs):
    #         if len(inputs) != len(inputs):
    #             raise ValueError("missing parameters")
    #
    #         # THIS IS CONSIDERABLY SLOWER THAN THE DYNAMIC FUNCTION ALTERNATIVE
    #         def compute(node):
    #             if node in other_inputs:
    #                 out = node.compute()
    #                 return out
    #             else:
    #                 ins = self.edges_in[node]
    #                 ins = map(lambda x: inputs[in_map[x]] if x in in_map else compute(x), ins)
    #                 out = node.compute(*ins)
    #                 return out
    #
    #         outs = tuple(map(compute, self.out_nodes))
    #         return outs
    #
    #     return compiled_graph

    def __call__(self, *input_values):
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
            input_values = [input_dict[input_layer] for input_layer in ord_inputs]

        elif len(input_values) > len(self.in_nodes):
            raise ValueError(f"too many inputs:\n"
                             f"\tgraph expects {len(self.in_nodes)} inputs\n"
                             f"\tinputs passed {len(input_values)}")

        ord_inputs = dict.fromkeys(list(self.in_nodes)[:len(input_values)])
        print(input_values)
        input_dict = dict(zip(ord_inputs.keys(), input_values))
        other_inputs = set(self.in_nodes).difference(ord_inputs)

        print(input_dict)

        def compute(node):
            if node in other_inputs:
                out = node.compute()
                return out
            else:
                ins = self.edges_in[node]
                ins = map(lambda x: input_dict[x] if x in input_dict else compute(x), ins)
                out = node.compute(*ins)
                return out

        return tuple(map(compute, self.out_nodes))


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
