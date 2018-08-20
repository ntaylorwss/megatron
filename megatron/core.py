import os
import numpy as np
import inspect
from . import utils


class Feature:
    def __init__(self, graph, name, n_dims):
        self.input_nodes = []
        self.output = None
        self.str = None
        self.graph = graph
        self.name = name
        self.n_dims = n_dims
        self.graph.add_feature(self)

    def validate_input(self, X):
        pass

    def __call__(self, observations):
        self.run(observations)
        self.graph.eager = True
        return self

    def __str__(self):
        if self.str is None:
            self.str = utils.md5_hash(self.output)
        return self.str

    def run(self, observations):
        self.validate_input(observations)
        self.output = observations


class Lambda:
    def __init__(self, function, **kwargs):
        self.function = function
        self.kwargs = kwargs

    def __call__(self, *inputs):
        return self.function(*inputs, **self.kwargs)

    def __str__(self):
        kw_values = [str(hp) for hp in self.kwargs.values()]
        return '{}({})'.format(self.__class__.__name__, ','.join(kw_values))


class Transformer:
    def __init__(self, **kwargs):
        self.metadata = utils.MetadataDict()
        self.kwargs = utils.KwargsDict(kwargs)
        self.is_fitted = False

    def __call__(self, *inputs):
        if not self.is_fitted:
            self.fit(*inputs)
            self.is_fitted = True
        return self.transform(*inputs)

    def __str__(self):
        hp_values = [str(hp) for hp in self.kwargs.values()]
        return '{}({})'.format(self.__class__.__name__, ','.join(hp_values))

    def fit(self, *inputs):
        pass

    def transform(self, *inputs):
        return inputs


class Transformation:
    def __init__(self, transformer):
        self.input_nodes = []
        self.output = None
        self.graph = None
        self.str = None
        self.transformer = transformer

    def _check_same_graph(self):
        if len(set([node.graph for node in self.input_nodes])) > 1:
            raise ValueError("Input nodes from different graphs provided to one transformation")

    def __str__(self):
       if not self.str:
           s = inspect.getsource(self.function)
           s += str(self.transformer)
           self.str = utils.md5_hash(s)
       return self.str

    def __call__(self, input_nodes):
        self.input_nodes = utils.listify(input_nodes)
        self._check_same_graph()

        self.graph = self.input_nodes[0].graph
        self.graph.add_transformation(self)

        if self.graph.eager:
            self.run()

        return self

    def run(self):
        inputs = [node.output for node in self.input_nodes]
        self.output = self.transformer(*inputs)


class Graph:
    def __init__(self, train_mode=True, cache_path='../cache'):
        self.train_mode = train_mode
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        self.eager = False
        self.features = set()
        self.transformations = set()

    def add_feature(self, feature):
        self.features.add(feature)

    def add_transformation(self, transformation):
        self.transformations.add(transformation)

    def _postorder_traversal(self, output_node):
        result = []
        if output_node:
            for child in output_node.input_nodes:
                result += self._postorder_traversal(child)
            result.append(output_node)
        return result

    def _run_path_with_caching(self, output_node, feed_dict):
        full_path = self._postorder_traversal(output_node)
        # run just the Feature nodes to get the data hashes
        node_index = 0
        while node_index < len(full_path) and isinstance(full_path[node_index], Feature):
            node = full_path[node_index]
            node.run(feed_dict[node.name])
            node_index += 1
        # skip to end, walk the path backwards looking for saves; if none, save this one
        node_index = len(full_path) - 1
        while full_path[node_index].output is None:
            path_hash = utils.md5_hash(''.join(str(node) for node in full_path[:node_index+1]))
            filepath = "{}/{}.npz".format(self.cache_path, path_hash)
            if os.path.exists(filepath):
                full_path[node_index].output = np.load(filepath)['arr']
                print("Loading node number {} in path from cache".format(node_index))
                break
            else:
                node_index -= 1
        # walk forwards again, running nodes to get output until reaching terminal
        while True:
            if isinstance(full_path[node_index], Transformation):
                full_path[node_index].run()
            if full_path[node_index] == output_node:
                break
            node_index += 1
        # cache full path as compressed file for future use, unless it already exists
        out = full_path[-1].output
        path_hash = utils.md5_hash(''.join(str(node) for node in full_path))
        if not os.path.exists(path_hash):
            np.savez_compressed('{}/{}.npz'.format(self.cache_path, path_hash), arr=out)
        # return the terminal node's resulting data
        return out

    def _run_path(self, output_node, feed_dict):
        full_path = self._postorder_traversal(output_node)
        for node in full_path:
            if isinstance(node, Feature):
                node.run(feed_dict[node.name])
            else:
                node.run()
        return full_path[-1].output

    def run(self, output_nodes, feed_dict, cache=True):
        if self.eager:
            raise utils.EagerRunException()
        out = []
        output_nodes = utils.listify(output_nodes)
        for output_node in output_nodes:
            if cache:
                out.append(self._run_path_with_caching(output_node, feed_dict))
            else:
                out.append(self._run_path(output_node, feed_dict))
        return out[0] if len(out) == 1 else out
