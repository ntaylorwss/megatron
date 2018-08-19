from .utils import listify


class Feature:
    def __init__(self, graph, name, n_dims):
        self.input_nodes = []
        self.output = None
        self.graph = graph
        self.name = name
        self.n_dims = n_dims
        self.graph.add_feature(self)

    def validate_input(self, X):
        pass

    def run(self, observations):
        self.validate_input(observations)
        self.output = observations


class Transformation:
    def __init__(self, function, **hyperparameters):
        self.function = function
        self.hyperparameters = hyperparameters
        self.input_nodes = None
        self.output = None
        self.graph = None

    def __call__(self, input_nodes):
        input_nodes = listify(input_nodes)
        if len(set([node.graph for node in input_nodes])) > 1:
            raise ValueError("Input nodes from different graphs provided to one transformation")

        self.input_nodes = input_nodes
        self.graph = input_nodes[0].graph
        self.graph.add_transformation(self)
        return self

    def run(self):
        if self.output is not None:
            return
        inputs = [node.output for node in self.input_nodes]
        self.output = self.function(*inputs, **self.hyperparameters)


class Graph:
    def __init__(self, eager=False):
        self.eager = eager
        self.features = {}
        self.transformations = set()

    @staticmethod
    def _postorder_traversal(output_node):
        result = []
        if output_node:
            for child in output_node.input_nodes:
                result += Graph._postorder_traversal(child)
            result.append(output_node)
        return result

    def add_feature(self, feature):
        self.features[feature.name] = feature

    def add_transformation(self, transformation):
        self.transformations.add(transformation)

    def run(self, output_nodes, feed_dict):
        def run_path(output_node):
            path = self._postorder_traversal(output_node)
            for node in path:
                if isinstance(node, Feature):
                    node.run(feed_dict[node.name])
                elif isinstance(node, Transformation):
                    node.run()
            return node.output

        if isinstance(output_nodes, list):
            out = list(map(run_path, output_nodes))
        else:
            out = run_path(output_nodes)
        return out

