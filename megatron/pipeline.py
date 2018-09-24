import os
import numpy as np
import pandas as pd
import dill as pickle
from . import utils
from .nodes import InputNode, TransformationNode


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
    def __init__(self, cache_dir='../feature_cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.eager = False
        self.nodes = []

    def _add_node(self, node):
        """Add a node to the pipeline.

        Parameters
        ----------
        node : TransformationNode / InputNode
            the node to be added, whether an InputNode or TransformationNode.
        """
        self.nodes.append(node)

    def _postorder_traversal(self, output_node):
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
        visited = set()
        order = []
        def dfs(node):
            visited.add(node)
            for in_node in node.input_nodes:
                if not in_node in visited:
                    dfs(in_node)
            order.append(node)
        dfs(output_node)
        return order

    def _run_path(self, path, feed_dict, cache_result):
        """Execute all non-cached nodes along the path given input data.

        Can cache the result for a path if requested.

        Parameters
        ----------
        path : list of TransformationNode
            the path of TransformationNodes to be executed.
        feed_dict : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.

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
                node.run(feed_dict[node.name])
                inputs_loaded += 1
            if inputs_loaded == num_inputs:
                break

        # walk the path backwards from the end, check each subpipeline for cached version
        # if none are cached, cache this one
        for node_index in range(len(path)-1, -1, -1):
            path_hash = utils.md5_hash(''.join(str(node) for node in path[:node_index+1]))
            filepath = "{}/{}.npz".format(self.cache_dir, path_hash)
            if os.path.exists(filepath):
                path[node_index].output = np.load(filepath)['arr']
                print("Loading node number {} in path from cache".format(node_index))
                # because this node has data, start from the next one
                node_index += 1
                break
        # walk forwards again, running nodes to get output until reaching terminal
        while node_index < len(path):
            if isinstance(path[node_index], TransformationNode):
                path[node_index].run()
            # clear data from nodes once they've been used by all output nodes
            for node in path[:node_index]:
                if (node.output is not None
                        and all(out_node.output is not None for out_node in node.output_nodes)):
                    node.output = None
            node_index += 1
        # optionally cache full path as compressed file for future use
        out = path[-1].output
        if cache_result:
            path_hash = utils.md5_hash(''.join(str(node) for node in path))
            filepath = '{}/{}.npz'.format(self.cache_dir, path_hash)
            if not os.path.exists(filepath):
                np.savez_compressed(filepath, arr=out)
        return out

    def _format_output(self, data, dtype, names):
        """Return data for single or multiple TransformationNode(s) in requested format.

        Parameters
        ----------
        data : list of np.ndarray
            data resulting from Pipeline.run(). Will always be a list, potentially of one.
        dtype : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        names : list of str
            names of output nodes; used when dtype is 'dataframe'.
        """
        if dtype == 'array':
            return data[0] if len(data) == 1 else data
        elif dtype == 'dataframe':
            if len(data) == 1:
                data = pd.DataFrame(data[0], columns=names)
            else:
                data = pd.DataFrame(np.stack(data).T, columns=names)
            return data
        else:
            raise ValueError("Invalid dtype; should be either 'array' or 'dataframe'")

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

    def run(self, output_nodes, feed_dict, cache_result=True, refit=False, dtype='array'):
        """Execute a path terminating at (a) given TransformationNode(s) with some input data.

        Parameters
        ----------
        output_nodes : list of TransformationNode
            the terminal nodes for which to return data.
        feed_dict : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        refit : bool
            applies to Transformation nodes; if True, recalculate metadata based on this data.
        dtype : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        """
        if isinstance(feed_dict, pd.DataFrame):
            feed_dict = dict(zip(feed_dict.columns, feed_dict.T.values))

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
        names = [node.name for node in output_nodes]
        return self._format_output(out, dtype, names)


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
