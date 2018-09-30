import os
import numpy as np
import pandas as pd
import dill as pickle
from collections import defaultdict
from .utils.generic import md5_hash, listify
from .utils.errors import EagerRunError, DisconnectedError
from .nodes import InputNode, TransformationNode
from .nodes.wrappers import FeatureSet
from .adapters import output


class Pipeline:
    """A pipeline with nodes as Transformations and InputNodes, edges as I/O relationships.

    Pipelines internally implement intelligent caching for maximal data re-use.
    Pipelines can also be saved with metadata intact for future use.

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
        when True, TransformationNode outputs are to be calculated on creation. This is indicated by
        data being passed to an InputNode node as a function call.
    nodes : list of TransformationNode / InputNode
        all InputNode and TransformationNode nodes belonging to the Pipeline.
    """
    def __init__(self, inputs, outputs, cache_dir='feature_cache'):
        self.inputs = []
        inputs = listify(inputs)
        for node in inputs:
            if isinstance(node, FeatureSet):
                self.inputs += node.nodes
            else:
                self.inputs.append(node)
        self.outputs = listify(outputs)
        self.path = self._topsort(self.outputs)
        not_provided = set(self.path).intersection(self.inputs) - set(self.inputs)
        if len(not_provided) > 0:
            raise DisconnectedError(not_provided)
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.eager = False
        self._tmp_inbound_storage = {}

    def _topsort(self, output_nodes):
        """Returns the path to the desired TransformationNode through the Pipeline.

        Parameters
        ----------
        output_node : TransformationNode
            the target terminal node of the path.

        Returns
        -------
        list of TransformationNode
            the path from input to output that arrives at the output_node.
        """
        visited = defaultdict(int)
        order = []
        def dfs(node):
            visited[node] += 1
            for in_node in node.inbound_nodes:
                if not in_node in visited:
                    dfs(in_node)
            if visited[node] <= 1:
                order.append(node)
        for output_node in output_nodes:
            dfs(output_node)
        return order

    def _run_path(self, input_data, cache_result):
        """Execute all non-cached nodes along the path given input data.

        Can cache the result for a path if requested.

        Parameters
        ----------
        path : list of TransformationNode
            the path of TransformationNodes to be executed.
        input_data : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy arrays in the cache.

        Returns
        -------
        np.ndarray
            the data for the target node of the path given the input data.
        """
        # path starts as entire path of graph
        path = self.path

        # run just the InputNode nodes to get the data hashes
        inputs_loaded = 0
        num_inputs = sum(1 for node in path if isinstance(node, InputNode))
        for node in path:
            if isinstance(node, InputNode):
                node.run(input_data[node.name])
                inputs_loaded += 1
            if inputs_loaded == num_inputs:
                break

        # find cached nodes and erase their inbounds, to create a shorter path
        cache_filepaths = {}
        for node in path:
            subpath = self._topsort([node])
            path_hash = md5_hash(''.join(str(node) for node in subpath))
            filepath = '{}/{}.npz'.format(self.cache_dir, path_hash)
            if os.path.exists(filepath):
                cache_filepaths[node] = filepath
                print("Loading node named '{}' from cache".format(node.name))
        for node, filepath in cache_filepaths.items():
            self._tmp_inbound_storage[node] = node.inbound_nodes
            node.inbound_nodes = []
            node.output = np.load(filepath)['arr']

        # recalculate path to outputs based on now removed edges
        path = self._topsort(self.outputs)

        # run transformation nodes to end of path
        for index, node in enumerate(path):
            if isinstance(node, TransformationNode):
                try:
                    if node.output is None:  # could be cache-loaded TransformationNode
                        node.run()
                except Error as e:
                    print("Error thrown at node named {}".format(node.name))
                    raise
            # erase data from nodes once unneeded
            for predecessor in path[:index]:
                if (predecessor.output is not None
                        and all(out_node.output is not None
                                for out_node in predecessor.outbound_nodes)
                        and predecessor not in self.outputs):
                    predecessor.output = None

        # reset inbound node tracking
        for node, inbound_nodes in self._tmp_inbound_storage.items():
            node.inbound_nodes = inbound_nodes
        self._tmp_inbound_storage = {}

        # cache results if asked
        if cache_result:
            for node in self.outputs:
                hashes = [str(node) for node in self._topsort([node])]
                path_hash = md5_hash(''.join(hashes))
                filepath = '{}/{}.npz'.format(self.cache_dir, path_hash)
                if not os.path.exists(filepath):
                    np.savez_compressed(filepath, arr=node.output)

    def _format_output(self, data, form, names):
        """Return data for single or multiple TransformationNode(s) in requested format.

        Parameters
        ----------
        data : list of np.ndarray
            data resulting from Pipeline.run(). Will always be a list, potentially of one.
        form : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        names : list of str
            names of output nodes; used when form is 'dataframe'.
        """
        if form== 'array':
            return output.array(data)
        elif form== 'dataframe':
            return output.dataframe(data, names)
        else:
            raise ValueError("Invalid format; should be either 'array' or 'dataframe'")

    def save(self, filepath):
        """Store just the nodes without their data (i.e. pre-execution) in a given file.

        Parameters
        ----------
        filepath : str
            the desired location of the stored nodes, filename included.
        """
        # TODO: make this more like Keras by outputting a JSON description of the model structure
        # store ref to data outside of Pipeline and remove ref to data in TransformationNodes
        data = {}
        for node in self.path:
            data[node] = node.output
            node.output = None
        with open(filepath, 'wb') as f:
            # keep same cache_dir too for new pipeline when loaded
            pipeline_info = {'nodes': self.path, 'cache_dir': self.cache_dir}
            pickle.dump(pipeline_info, f)
        # reinsert data into Pipeline
        for node in self.path:
            node.output = data[node]

    def set_outputs(self, new_outputs):
        self.outputs = listify(new_outputs)
        self.path = self._topsort(self.outputs)

    def transform(self, input_data, cache_result=True, refit=False, form='array'):
        """Execute a path terminating at (a) given TransformationNode(s) with some input data.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        refit : bool
            applies to Transformation nodes; if True, recalculate metadata based on this data.
        form : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        """
        if self.eager:
            raise EagerRunError()

        if isinstance(input_data, pd.DataFrame):
            input_data = dict(zip(input_data.columns, input_data.T.values))

        out = []
        if refit:
            for node in self.path:
                node.is_fitted = False
        self._run_path(input_data, cache_result)
        for node in self.outputs:
            out.append(node.output)
        names = [node.name for node in self.outputs]
        return self._format_output(out, form, names)

    def transform_generator(self, input_generators, cache_result=True, refit=False, form='array'):
        def dict_to_generator(d):
            keys = d.keys()
            for values in zip(*d.values()):
                yield dict(zip(keys, values))

        input_generator = dict_to_generator(input_generators)
        for input_data in input_generator:
            yield self.transform(input_data, cache_result, refit, form)


def load_pipeline(filepath):
    """Load a set of nodes from a given file, stored previously with Pipeline.save().

    Parameters
    ----------
    filepath : str
        the file from which to load a Pipeline.
    """
    with open(filepath, 'rb') as f:
        pipeline_info = pickle.load(f)
    G = Pipeline(cache_dir=pipeline_info['cache_dir'])
    for node in pipeline_info['nodes']:
        G._add_node(node)
    return G
