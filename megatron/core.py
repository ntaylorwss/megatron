import os
import numpy as np
import inspect
from . import utils


class Node:
    def __init__(self, transformation, input_nodes):
        self.transformation = transformation
        self.graph = input_nodes[0].graph
        self.graph.add_node(self)
        self.input_nodes = input_nodes
        self.output = None
        self.is_fitted = False

    def run(self):
        inputs = [node.output for node in self.input_nodes]
        if not self.is_fitted:
            self.transformation.fit(*inputs)
            self.is_fitted = True
        self.output = self.transformation.transform(*inputs)

    def __str__(self):
        return str(self.transformation)


class Input:
    def __init__(self, graph, name, input_shape=(1,)):
        self.graph = graph
        self.graph.add_node(self)
        self.name = name
        self.input_nodes = []
        self.input_shape = input_shape
        self.str = None
        self.output = None
        self.is_fitted = False

    def validate_input(self, X):
        if list(X.shape[1:]) != list(self.input_shape):
            raise utils.ShapeError(self.name, self.input_shape, X.shape[1:])

    def run(self, observations):
        self.validate_input(observations)
        self.output = observations

    def __call__(self, observations):
        self.run(observations)
        self.graph.eager = True
        return self

    def __str__(self):
        if self.str is None:
            self.str = utils.md5_hash(self.output)
        return self.str


class Lambda:
    def __init__(self, transform_fn, **kwargs):
        self.transform_fn = transform_fn
        self.kwargs = kwargs

    def __call__(self, *inputs):
        node = Node(self, input_nodes)
        if node.graph.eager:
            node.run()
        return node

    def __str__(self):
        out = [str(hp) for hp in self.kwargs.values()]
        out.append(inspect.getsource(self.transform))
        return ''.join(out)

    def fit(self, *inputs):
        pass

    def transform(self, *inputs):
        return self.transform_fn(*inputs)


class Transformation:
    def __init__(self):
        self.metadata = utils.MetadataDict()
        self.is_fitted = False

    def __call__(self, *input_nodes):
        node = Node(self, input_nodes)
        if node.graph.eager:
            node.run()
        return node

    def __str__(self):
        metadata = ''.join([utils.md5_hash(metadata) for metadata in self.metadata.values()])
        return '{}{}'.format(inspect.getsource(self.transform), metadata)

    def fit(self, *inputs):
        pass

    def transform(self, *inputs):
        return inputs


class Graph:
    def __init__(self, train_mode=True, cache_dir='../feature_cache'):
        self.train_mode = train_mode
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.eager = False
        self.nodes = set()

    def add_node(self, node):
        self.nodes.add(node)

    def _postorder_traversal(self, output_node):
        result = []
        if output_node:
            for child in output_node.input_nodes:
                result += self._postorder_traversal(child)
            result.append(output_node)
        return result

    def _run_path(self, path, feed_dict, cache_result):
        # run just the Input nodes to get the data hashes
        inputs_loaded = 0
        num_inputs = sum(1 for node in self.nodes if isinstance(node, Input))
        for node in path:
            if isinstance(node, Input):
                node.run(feed_dict[node.name])
                inputs_loaded += 1
            if inputs_loaded == num_inputs:
                break

        # walk the path backwards from the end, check each subgraph for cached version
        # if none are cached, cache this one
        for node_index in range(len(path)-1, -1, -1):
            path_hash = utils.md5_hash(''.join(str(node) for node in path[:node_index+1]))
            filepath = "{}/{}.npz".format(self.cache_dir, path_hash)
            if os.path.exists(filepath):
                path[node_index].output = np.load(filepath)['arr']
                print("Loading node number {} in path from cache".format(node_index))
                break
        # walk forwards again, running nodes to get output until reaching terminal
        while True:
            if isinstance(path[node_index], Node):
                path[node_index].run()
            if node_index == len(path) - 1:
                break
            node_index += 1
        # optionally cache full path as compressed file for future use
        out = path[-1].output
        if cache_result:
            path_hash = utils.md5_hash(''.join(str(node) for node in path))
            filepath = '{}/{}.npz'.format(self.cache_dir, path_hash)
            if not os.path.exists(filepath):
                np.savez_compressed(filepath, arr=out)
        return out

    def save(self, filepath):
        # store ref to data, None the ref in the Node
        data = {}
        for node in self.nodes:
            data[node] = node.output
            node.output = None
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        for node in nodes:
            node.output = data[node]

    def run(self, output_nodes, feed_dict, cache_result=True, refit=False):
        if self.eager:
            raise utils.EagerRunException()
        out = []
        output_nodes = utils.listify(output_nodes)
        for output_node in output_nodes:
            path = self._postorder_traversal(output_node)
            if refit:
                for node in path:
                    node.transformation.is_fitted = False
            out.append(self._run_path(path, feed_dict, cache_result))
        return out[0] if len(out) == 1 else out


def load_graph(filepath):
    with open(filepath, 'rb') as f:
        out = pickle.load(f)
    return out
