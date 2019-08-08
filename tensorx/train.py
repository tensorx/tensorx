from tensorx.utils import as_list, Graph
import tensorflow as tf
import tensorx as tx


class LayerGraph:
    def __init__(self, outputs, inputs=None, other_tensors=None, other_inputs=None):
        self.other_tensors = as_list(other_tensors)
        self.other_inputs = as_list(other_inputs)
        self.input_layers = as_list(inputs)
        self.output_layers = as_list(outputs)

        dependencies = {}

        for output in self.output_layers:
            dependencies[output] = []

        visited = set()
        missing_dep = {}
        global_graph = Graph()

        for output in self.output_layers:
            graph = Graph.build_graph(input_layers=None, output_layers=output)
            for dependency in graph.in_nodes:
                if dependency not in self.input_layers and len(self.input_layers) != 0:
                    if output not in missing_dep:
                        missing_dep[output] = []
                    if dependency not in missing_dep[output]:
                        missing_dep[output].append(dependency)
            global_graph = Graph.merge(global_graph, graph)

        if len(missing_dep) > 0:
            raise ValueError("Could not create graph: \n Missing input dependencies:"
                             "\n {missing}".format(missing="\n".join([
                str(o) + "<---[{}]".format(",".join(
                    map(str, i))) for o, i in missing_dep.items()
            ])))

        self.graph = global_graph
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

    def eval(self, feed=None,
             other_tensors=None,
             target_outputs=None,
             options=None,
             run_metadata=None):
        """ Evaluates the current graph on the given inputs

        if input_values are used and Inputs have values != None, these are not overwritten
        if a feed dictionary with layer-->data is passed, only the missing inputs are possibly
        fed with their default values.

        Args:
            target_outputs:
            other_tensors: runs other tensors or ops that might not be included in the graph
            feed: a feed dictionary from Input Layers or Parameters to values, if None, matches the
            inputs with self.inputs in the same order. If default values are available in the input
            layers, these are used instead.

            options: A [RunOptions] protocol buffer
            run_metadata: A [RunMetadata] protocol buffer

        Returns:
            the result of the graph evaluation

        """
        if feed is None:
            feed = {}

        if not isinstance(feed, dict):
            raise TypeError("feed must be a dictionary from inputs to values")

        target_outputs = as_list(target_outputs)

        output_layers = self.output_layers
        other_tensors = self.other_tensors + as_list(other_tensors)

        if len(target_outputs) > 0:
            invalid_out = [target for target in target_outputs if target not in self.output_layers]
            if len(invalid_out) != 0:
                raise ValueError("Invalid target outputs. outputs not in the graph:\n"
                                 "{outs}".format(outs="\n".join(map(str, invalid_out))))
            output_layers = target_outputs

        # feed the values to the dynamic tensor inputs
        for in_layer, data in feed.items():
            if in_layer in self.graph.in_nodes:
                in_layer.value = data

        outputs = output_layers + other_tensors

        @tf.function
        def compute():
            return tuple([output.compute() if isinstance(output, tx.Layer) else output for output in outputs])

        result = compute()
        if len(output_layers) == 1 and len(other_tensors) == 0:
            result = result[0]

        result = [result.numpy() for result in result]
        return result
