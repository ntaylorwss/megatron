import os
import numpy as np
import inspect
from collections import defaultdict
from . import utils


class Node:
    """A graph node holding a Transformation.

    It connects to a set of input Nodes (of class Node or Input) and, when run,
    applies its given Transformation, storing the result in its output variable.

    Parameters
    ----------
    transformation : megatron.Transformation
        the transformation to be applied to the data from its input Nodes.
    input_nodes : list of megatron.Node / megatron.Input
        the Nodes to be connected to this node as input.

    Attributes
    ----------
    transformation : megatron.Transformation
        the transformation to be applied to the data from its input Nodes.
    graph : megatron.Graph
        the Graph with which this Node is associated; deduced from input nodes.
    output : None or np.ndarray
        is None until Node is run; when run, the Numpy array produced is stored here.
    is_fitted : bool
        indicates whether the Transformation inside the Node
        has, if necessary, been fit to data.
    """
    def __init__(self, transformation, input_nodes, name=None):
        self.transformation = transformation
        self.graph = input_nodes[0].graph
        self.graph._add_node(self, name)
        self.input_nodes = input_nodes
        self.output = None
        self.is_fitted = False

    def run(self):
        """Stores result of given Transformation on input Nodes in output variable."""
        inputs = [node.output for node in self.input_nodes]
        if not self.is_fitted:
            self.transformation.fit(*inputs)
            self.is_fitted = True
        self.output = self.transformation.transform(*inputs)

    def __str__(self):
        """Used in caching subgraphs."""
        return str(self.transformation)


class Input:
    """A graph node that is fed input data as Numpy arrays.

    It is always an initial node in a Graph (no input nodes) and, when run,
    stores its given data (either from a feed dict or a function call) in its output.

    Parameters
    ----------
    graph : megatron.Graph
        the Graph with which the node is associated.
    name : str
        a name to associate with the data; the keys of the Graph feed dict will be these names.
    input_shape : tuple of int
        the shape, not including the observation dimension (1st), of the Numpy arrays to be input.

    Attributes
    ----------
    graph : megatron.Graph
        the Graph with which the node is associated.
    name : str
        a name to associate with the data; the keys of the Graph feed dict will be these names.
    input_shape : tuple of int
        the shape, not including the observation dimension (1st), of the Numpy arrays to be input.
    str : str
        stores the result of magic method str, so that it doesn't have to be recalculated.
    output : np.ndarray
        is None until node is run; when run, the Numpy array passed in is stored here.
    """
    def __init__(self, graph, name, input_shape=(1,)):
        self.graph = graph
        self.graph._add_node(self, name)
        self.name = name
        self.input_nodes = []
        self.input_shape = input_shape
        self.str = None
        self.output = None

    def validate_input(self, observations):
        """Ensure shape of data passed in aligns with shape of the node.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        if hasattr(observations, 'shape') and (list(observations.shape[1:]) != list(self.input_shape)):
            raise utils.ShapeError(self.name, self.input_shape, observations.shape[1:])

    def run(self, observations):
        """Validate and store the data passed in.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        self.validate_input(observations)
        self.output = observations

    def __call__(self, observations):
        """Run the node, and indicate to the associated Graph that it is running eagerly.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        self.run(observations)
        self.graph.eager = True
        return self

    def __str__(self):
        """Used in caching subgraphs."""
        if self.str is None:
            self.str = utils.md5_hash(self.output)
        return self.str


class Lambda:
    """A wrapper for a stateless custom function.

    For custom functions that are stateless, and thus do not require to be fit,
    a Lambda wrapper is preferred to creating a Transformation subclass.

    Parameters
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    name : str or None
        optional custom name for the node. If none given, default name is
        [name of transform_fn]:[index], where index is a unique identifier for
        multiple nodes with the same transform_fn.
    **kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.

    Attributes
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    name : str or None
        optional custom name for the node. If none given, default name is
        [name of transform_fn]:[index], where index is a unique identifier for
        multiple nodes with the same transform_fn.
    **kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.
    """
    def __init__(self, transform_fn, name=None, **kwargs):
        self.transform_fn = transform_fn
        self.kwargs = kwargs
        if name:
            self.name = name
        else:
            self.name = transform_fn.__name__

    def __call__(self, *input_nodes):
        """Creates a Node associated with this Lambda Transformation and the given input Nodes.

        Parameters
        ----------
        *input_nodes : megatron.Node(s) / megatron.Input(s)
            the input nodes, whose data are to be passed to transform_fn when run.
        """
        node = Node(self, input_nodes, self.name)
        if node.graph.eager:
            node.run()
        return node

    def __str__(self):
        """Used in caching subgraphs."""
        out = [str(hp) for hp in self.kwargs.values()]
        out.append(inspect.getsource(self.transform))
        return ''.join(out)

    def fit(self, *inputs):
        """Does nothing; only defined for compliance with the Transformation interface."""
        pass

    def transform(self, *inputs):
        """Apply transform_fn to given input data.

        Parameters
        ----------
        inputs : np.ndarray(s)
            input data to be passed to transform_fn.
        """
        return self.transform_fn(*inputs)


class Transformation:
    """Base class of a potentially stateful function to be used to transform data.

    For custom functions that are stateful, and thus require to be fit,
    writing a Transformation subclass is required rather than using a Lambda wrapper.

    Parameters
    ----------
    name : str or None
        optional custom name for the node. If none given, default name is
        [name of class]:[index], where index is a unique identifier for
        multiple nodes of the same object.

    Attributes
    ----------
    metadata : dict
        stores any necessary metadata, which is defined by child class.
    """
    def __init__(self, name=None):
        self.metadata = {}
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def __call__(self, *input_nodes):
        """Creates a Node associated with this Transformation and the given input Nodes.

        Parameters
        ----------
        *input_nodes : megatron.Node(s) / megatron.Input(s)
            the input nodes, whose data are to be passed to transform_fn when run.
        """
        node = Node(self, input_nodes, self.name)
        if node.graph.eager:
            node.run()
        return node

    def __str__(self):
        """Used in caching subgraphs."""
        metadata = ''.join([utils.md5_hash(metadata) for metadata in self.metadata.values()])
        return '{}{}'.format(inspect.getsource(self.transform), metadata)

    def fit(self, *inputs):
        """Calculates and overwrites metadata based on current inputs.

        Parameters
        ----------
        *inputs : numpy.ndarray(s)
            the input data to be fit to.
        """
        pass

    def transform(self, *inputs):
        """Apply transformation to given input data.

        Parameters
        ----------
        inputs : np.ndarray(s)
            input data to be transformed.
        """
        return inputs


class Graph:
    """A graph with nodes as Transformations and Inputs, edges as I/O relationships.

    Graphs internally implement intelligent caching for maximal data re-use.
    Graphs can also be saved with metadata intact for future use.

    Parameters
    ----------
    cache_dir : str (default: '../feature_cache')
        the relative path from the current working directory to store numpy data results
        for particular executions of nodes.

    Attributes
    ----------
    cache_dir : str
        the relative path from the current working directory to store numpy data results
        for particular executions of nodes.
    eager : bool
        when True, Node outputs are to be calculated on creation. This is indicated by
        data being passed to an Input node as a function call.
    nodes : list of Node / Input
        all Input and Node nodes belonging to the Graph.
    """
    def __init__(self, cache_dir='../feature_cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.eager = False
        self.nodes = defaultdict(list)

    def _add_node(self, node, name):
        """Add a node to the graph.

        Parameters
        ----------
        node : Node / Input
            the node to be added, whether an Input or Node.
        name : str
            the name of the node to be added.
        """
        self.nodes[name].append(node)

    def _lookup_node(self, node):
        """Get node by name as string, or just give back the node itself.

        Parameters
        ----------
        node : Node or str
            if Node, return node; if str, lookup node associated with str.
            str format should be '[transformation/input name]:[index].
            if the same transformation/input name is used more than once, subsequent instances
            are made unique by their index.

        Returns
        -------
        Node
            either the node given as an argument, or the node associated with the string argument.
        """
        if isinstance(node, str):
            if node.find(':') >= 0:
                name, index = node.split(':')
            else:
                name = node
                index = '0'
            return self.nodes[name][int(index)]
        else:
            return node

    def _postorder_traversal(self, output_node):
        """Returns the path to the desired Node through the Graph.

        Parameters
        ----------
        output_node : Node
            the target terminal node of the path.

        Returns
        -------
        list of Node
            the path from input to output that arrives at the output_node.
        """
        output_node = self._lookup_node(output_node)
        path = []
        if output_node:
            for child in output_node.input_nodes:
                path += self._postorder_traversal(child)
            path.append(output_node)
        return path

    def _run_path(self, path, feed_dict, cache_result):
        """Execute all non-cached nodes along the path given input data.

        Can cache the result for a path if requested.

        Parameters
        ----------
        path : list of Node
            the path of Nodes to be executed.
        feed_dict : dict of Numpy array
            the input data to be passed to Input Nodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.

        Returns
        -------
        np.ndarray
            the data for the target node of the path given the input data.
        """
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
        """Store just the nodes without their data (i.e. pre-execution) in a given file.

        Parameters
        ----------
        filepath : str
            the desired location of the stored nodes, filename included.
        """
        # store ref to data outside of Graph and remove ref to data in Nodes
        data = {}
        for node in self.nodes:
            data[node] = node.output
            node.output = None
            node.graph = None
        with open(filepath, 'wb') as f:
            pickle.dump(self.nodes, f)
        # reinsert data into Graph
        for node in nodes:
            node.output = data[node]

    def run(self, output_nodes, feed_dict, cache_result=True, refit=False):
        """Execute a path terminating at (a) given Node(s) with some input data.

        Parameters
        ----------
        output_nodes : list of Node or str
            the terminal nodes, or node names, for which to return data.
        feed_dict : dict of Numpy array
            the input data to be passed to Input Nodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        refit : bool
            applies to Transformation nodes; if True, recalculate metadata based on this data.
        """
        if self.eager:
            raise utils.EagerRunException()
        out = []
        output_nodes = utils.listify(output_nodes)
        for output_node in output_nodes:
            path = self._postorder_traversal(output_node)
            if refit:
                for node in path:
                    node.is_fitted = False
            out.append(self._run_path(path, feed_dict, cache_result))
        return out[0] if len(out) == 1 else out


def load_graph(filepath):
    """Load a set of nodes from a given file, stored previously with Graph.save().

    Parameters
    ----------
    filepath : str
        the file from which to load a Graph.
    """
    with open(filepath, 'rb') as f:
        nodes = pickle.load(f)
    self.nodes = nodes
