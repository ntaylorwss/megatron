import os
import numpy as np
import pandas as pd
import dill as pickle
from collections import defaultdict
from .utils.generic import md5_hash, listify
from .utils.errors import EagerRunException
from .nodes import InputNode, TransformationNode
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
    def __init__(self, cache_dir='feature_cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.eager = False
        self.nodes = []
        self._tmp_inbound_storage = {}

    def _add_node(self, node):
        """Add a node to the pipeline.

        Parameters
        ----------
        node : TransformationNode / InputNode
            the node to be added, whether an InputNode or TransformationNode.
        """
        self.nodes.append(node)

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

    def _run_path(self, path, output_nodes, input_data, cache_result):
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
        # run just the InputNode nodes to get the data hashes
        inputs_loaded = 0
        num_inputs = sum(1 for node in self.nodes if isinstance(node, InputNode))
        for node in path:
            if isinstance(node, InputNode):
                node.run(input_data[node.name])
                inputs_loaded += 1
            if inputs_loaded == num_inputs:
                break

        # find cached nodes and erase their inputs, so as to make them inputs
        cache_filepaths = {}
        for node in self.nodes:
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

        # recalculate path based on now removed edges
        path = self._topsort(output_nodes)

        # run transformation nodes to end of path
        for index, node in enumerate(path):
            if isinstance(node, TransformationNode):
                try:
                    if node.output is None:  # could be cache-loaded TransformationNode
                        node.run()
                except Exception as e:
                    print("Exception thrown at node named {}".format(node.name))
                    raise
            # erase data from nodes once unneeded
            for predecessor in path[:index]:
                if (predecessor.output is not None
                        and all(out_node.output is not None
                                for out_node in predecessor.outbound_nodes)
                        and predecessor not in output_nodes):
                    predecessor.output = None

        # reset inbound node tracking
        for node, inbound_nodes in self._tmp_inbound_storage.items():
            node.inbound_nodes = inbound_nodes
        self._tmp_inbound_storage = {}

        # cache results if asked
        if cache_result:
            for node in output_nodes:
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
        # store ref to data outside of Pipeline and remove ref to data in TransformationNodes
        data = {}
        for node in self.nodes:
            data[node] = node.output
            node.output = None
            node.pipeline = None
        with open(filepath, 'wb') as f:
            # keep same cache_dir too for new pipeline when loaded
            pipeline_info = {'nodes': self.nodes, 'cache_dir': self.cache_dir}
            pickle.dump(pipeline_info, f)
        # reinsert data into Pipeline
        for node in self.nodes:
            node.output = data[node]

    def transform(self, output_nodes, input_data, cache_result=True, refit=False, form='array'):
        """Execute a path terminating at (a) given TransformationNode(s) with some input data.

        Parameters
        ----------
        output_nodes : list of TransformationNode
            the terminal nodes for which to return data.
        input_data : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        refit : bool
            applies to Transformation nodes; if True, recalculate metadata based on this data.
        form : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        """
        if isinstance(input_data, pd.DataFrame):
            input_data = dict(zip(input_data.columns, input_data.T.values))

        if self.eager:
            raise EagerRunException()

        out = []
        output_nodes = listify(output_nodes)
        path = self._topsort(output_nodes)
        if refit:
            for node in path:
                node.is_fitted = False
        self._run_path(path, output_nodes, input_data, cache_result)
        for node in output_nodes:
            out.append(node.output)
        names = [node.name for node in output_nodes]
        return self._format_output(out, form, names)


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
