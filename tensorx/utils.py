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
        in_nodes(set): a set with all the unique input nodes of the graph (nodes without edges coming into them)
        out_nodes(set): a set with all the unique output nodes of the graph (nodes without edges coming out of them)

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
        self.in_nodes = set()
        self.out_nodes = set()

        self.edges_in = dict()
        self.edges_out = dict()

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges_in[node] = []
            self.edges_out[node] = []
            self.in_nodes.add(node)
            self.out_nodes.add(node)

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        self.edges_out[node1].append(node2)
        self.edges_in[node2].append(node1)

        # update endpoints
        if node1 in self.out_nodes:
            self.out_nodes.remove(node1)
        if node2 in self.in_nodes:
            self.in_nodes.remove(node2)

    @staticmethod
    def build_graph(input_layers, output_layers):
        """ build_graph

        Args:
            input_layers: input terminal layers where the graph must stop
            output_layers: output layers from which we start to populate the graph

        Returns:
            a graph from the output layers to the given input layers
        """
        graph = Graph()
        input_layers = as_list(input_layers)
        output_layers = as_list(output_layers)

        # add terminals to the graph
        # for layer in input_layers:
        #    graph.add_node(layer)
        for layer in output_layers:
            graph.add_node(layer)

        visited = set()

        def _update_graph(current_layer):
            """

            Args:
                current_layer: output layer from which we start to populate the graph backwards using the input_layers

            Returns:
                True if a path was found between the output node and a given input layer
            """
            if current_layer in visited:
                return True
            else:
                visited.add(current_layer)
            if len(input_layers) > 0:
                if current_layer in input_layers:
                    return True
            elif len(current_layer.input_layers) == 0:
                return True

            path_found = {l: _update_graph(l) for l in current_layer.input_layers}
            found = False
            for input_layer in path_found.keys():
                found = path_found[input_layer] or True
                graph.add_edge(input_layer, current_layer)

            return found

        paths_found = [_update_graph(out_layer) for out_layer in output_layers]
        if not all(paths_found):
            failed = [str(output_layers[i]) for i, path_found in enumerate(paths_found) if not path_found]
            failed_layers = "\n".join(failed)
            raise ValueError("no path found between input and output layers: \n {}".format(failed_layers))

        if len(input_layers) > 0:
            not_found = [layer for layer in input_layers if layer not in graph.in_nodes]

            if len(not_found) > 0:
                raise ValueError("there is not path between the output layers and the following input layers: \n\t"
                                 "{}".format("\n\t".join(map(str, not_found))))

        return graph

    def compile_graph(self):
        """ compiles the graph into a tensorflow callable compiled graph

        the idea is to use exec to create a function and then call tf.function
        on the created function. this is to avoid using loops and lists to
        run through the Graph instance and call compute.

        function parameters:
            converts all the non-constant TensorLayer nodes into graph arguments

        run through each layer and queue up nodes starting on the input
        write down a function as a series of compute calls with inputs from the previous
        layer outputs

        Returns:
            a callable tensorflow graph

        """
        # TODO implement this
        raise NotImplementedError()


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
    if a list or tuple of elements is passed, returns a list with the elements

    Note: we exclude SparseTensorValue because it is a named tuple
    and we want to feed the whole object as a single data sample if needed

    Args:
        items: one item, a tuple of elements or a list of elements

    Returns:
        a :obj:`list` with the elements in items
    """
    if items is None:
        items = []
    elif isinstance(items, (list, tuple)) and not isinstance(items, (
            tf.compat.v1.SparseTensorValue, tf.SparseTensor)):
        items = list(items)
    else:
        items = [items]
    return items
