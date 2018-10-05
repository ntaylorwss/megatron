import os
import numpy as np
import pandas as pd
import dill as pickle
from collections import defaultdict
from . import utils


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
    def __init__(self, inputs, outputs):
        self.inputs = []
        inputs = utils.generic.listify(inputs)
        for node in inputs:
            if utils.generic.isinstance_str(node, 'FeatureSet'):
                self.inputs += node.nodes
            else:
                self.inputs.append(node)
        self.outputs = utils.generic.listify(outputs)
        self.path = utils.pipeline.topsort(self.outputs)
        not_provided = set(self.path).intersection(self.inputs) - set(self.inputs)
        if len(not_provided) > 0:
            raise utils.errors.DisconnectedError(not_provided)
        self.cache_dir = 'feature_cache'
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.eager = False
        self._tmp_inbound_storage = {}

    def _reload(self):
        for node in self.path:
            node.has_run = False

    def _load_inputs(self, input_data):
        inputs_loaded = 0
        num_inputs = sum(1 for node in self.path if utils.generic.isinstance_str(node, 'InputNode'))
        for node in self.path:
            if utils.generic.isinstance_str(node, 'InputNode'):
                node.load(input_data[node.name])
                inputs_loaded += 1
            if inputs_loaded == num_inputs:
                break

    def _fit(self, input_data, partial):
        self._reload()
        self._load_inputs(input_data)
        for index, node in enumerate(self.path):
            if utils.generic.isinstance_str(node, 'TransformationNode'):
                try:
                    if partial:
                        node.partial_fit()
                    else:
                        node.fit()
                except Exception as e:
                    print("Error thrown at node named {}".format(node.name))
                    raise
            # erase data from nodes once unneeded (including output nodes)
            for predecessor in self.path[:index]:
                if all(out_node.has_run for out_node in predecessor.outbound_nodes):
                    predecessor.output = None
        # erase last node
        self.path[-1].output = None
        # restore has_run
        for node in self.path:
            node.has_run = False

    def _transform(self, input_data, cache_result):
        """Execute all non-cached nodes along the path given input data.

        Can cache the result for a path if requested.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy arrays in the cache.

        Returns
        -------
        np.ndarray
            the data for the target node of the path given the input data.
        """
        self._reload()
        path = self.path

        # run just the InputNode nodes to get the data hashes
        self._load_inputs(input_data)

        # find cached nodes and erase their inbounds, to create a shorter path
        cache_filepaths = {}
        for node in path:
            subpath = utils.pipeline.topsort([node])
            path_hash = utils.hash.hash_path(subpath)
            filepath = '{}/{}.npz'.format(self.cache_dir, path_hash)
            if os.path.exists(filepath):
                cache_filepaths[node] = filepath
                print("Loading node named '{}' from cache".format(node.name))
        for node, filepath in cache_filepaths.items():
            self._tmp_inbound_storage[node] = node.inbound_nodes
            node.inbound_nodes = []
            node.output = np.load(filepath)['arr']

        # recalculate path to outputs based on now removed edges
        path = utils.pipeline.topsort(self.outputs)

        # run transformation nodes to end of path
        for index, node in enumerate(path):
            if utils.generic.isinstance_str(node, 'TransformationNode'):
                try:
                    if node.output is None:  # could be cache-loaded TransformationNode
                        node.transform()
                except Exception as e:
                    print("Error thrown at node named {}".format(node.name))
                    raise
            # erase data from nodes once unneeded
            for predecessor in path[:index]:
                outbound_run = all(out_node.has_run for out_node in predecessor.outbound_nodes)
                if outbound_run and predecessor not in self.outputs:
                    predecessor.output = None

        # reset inbound node tracking
        for node, inbound_nodes in self._tmp_inbound_storage.items():
            node.inbound_nodes = inbound_nodes
        self._tmp_inbound_storage = {}

        # cache results if asked
        if cache_result:
            for node in self.outputs:
                path_hash = utils.hash.hash_path(utils.pipeline.topsort([node]))
                filepath = '{}/{}.npz'.format(self.cache_dir, path_hash)
                if not os.path.exists(filepath):
                    np.savez_compressed(filepath, arr=node.output)

    def partial_fit(self, input_data):
        self._fit(input_data, True)

    def fit(self, input_data):
        self._fit(input_data, False)

    def fit_generator(self, input_generator):
        for batch in input_generator:
            self.partial_fit(batch)

    def transform(self, input_data, cache_result=True, form='array'):
        """Execute a path terminating at (a) given TransformationNode(s) with some input data.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        form : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        """
        if self.eager:
            raise utils.errors.EagerRunError()

        arrays = []
        self._transform(input_data, cache_result)
        for node in self.outputs:
            arrays.append(node.output)
            node.output = None
        names = [node.name for node in self.outputs]
        return utils.pipeline.format_output(arrays, form, names)

    def transform_generator(self, input_generator, cache_result=True, form='array'):
        for batch in input_generator:
            #yield batch
            yield self.transform(batch, cache_result, form)

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
