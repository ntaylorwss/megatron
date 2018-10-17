import os
import sqlite3
import numpy as np
import pandas as pd
import dill as pickle
from collections import defaultdict
from . import utils
from . import io


class Pipeline:
    """A pipeline with nodes as Transformations and InputNodes, edges as I/O relationships.

    Pipelines internally implement intelligent caching for maximal data re-use.
    Pipelines can also be saved with metadata intact for future use.

    Parameters
    ----------
    inputs : list of megatron.Node(s)
        input nodes of the pipeline, where raw data is fed in.
    outputs : list of megatron.Node(s)
        output nodes of the pipeline, the processed features.
    name : str
        unique identifying name of the pipeline.
    storage_db : Connection (defeault: 'sqlite')
        database connection to be used for input and output data storage.

    Attributes
    ----------
    inputs : list of megatron.Node(s)
        input nodes of the pipeline, where raw data is fed in.
    outputs : list of megatron.Node(s)
        output nodes of the pipeline, the processed features.
    path : list of megatron.Nodes
        full topological sort of pipeline from inputs to outputs.
    eager : bool
        when True, TransformationNode outputs are to be calculated on creation. This is indicated by
        data being passed to an InputNode node as a function call.
    nodes : list of TransformationNode / InputNode
        all InputNode and TransformationNode nodes belonging to the Pipeline.
    name : str
        unique identifying name of the pipeline.
    storage: Connection (defeault: None)
        storage database for input and output data.
    """
    def __init__(self, inputs, outputs, name,
                 version=None, storage=None):
        self.eager = False

        # flatten inputs into list of nodes
        self.inputs = []
        inputs = utils.generic.listify(inputs)
        for node in inputs:
            if utils.generic.isinstance_str(node, 'FeatureSet'):
                self.inputs += node.nodes
            elif utils.generic.isinstance_str(node, 'InputNode'):
                self.inputs.append(node)
            else:
                raise ValueError("Node provided as input that is not an InputNode")

        # flatten outputs into list of nodes
        self.outputs = []
        outputs = utils.generic.listify(outputs)
        for node in outputs:
            if utils.generic.isinstance_str(node, 'FeatureSet'):
                self.outputs += node.nodes
            else:
                self.outputs.append(node)

        # ensure all outputs are named
        if any(node.is_default_name for node in self.outputs):
            msg = "All outputs must be named; passed as second parameter when Layer is called"
            raise NameError(msg)

        # calculate path from input to output
        self.path = utils.pipeline.topsort(self.outputs)

        # ensure input data matches with input nodes
        missing_inputs = set(self.path).intersection(self.inputs) - set(self.inputs)
        if len(missing_inputs) > 0:
            raise utils.errors.DisconnectedError(missing_inputs)
        extra_inputs = set(self.path).intersection(self.inputs) - set(self.path)
        if len(extra_inputs) > 0:
            utils.errors.ExtraInputsWarning(extra_inputs)

        # setup output data storage
        self.name = name
        self.version = version
        if self.version:
            version = str(self.version).replace('.', '_')
        if storage:
            self.storage = io.storage.DataStore(self.name, version, storage)
        else:
            self.storage = None

    def _reload(self):
        """Reset all nodes' has_run indicators to False."""
        for node in self.path:
            node.has_run = False

    def _load_inputs(self, input_data):
        """Load data into Input nodes.

        Parameters
        ----------
        input_data : dict of np.ndarray
            dict mapping input names to input data arrays.
        """
        inputs_loaded = 0
        num_inputs = sum(1 for node in self.path if utils.generic.isinstance_str(node, 'InputNode'))
        for node in self.path:
            if utils.generic.isinstance_str(node, 'InputNode'):
                node.load(input_data[node.name])
                inputs_loaded += 1
            if inputs_loaded == num_inputs:
                break

    def _fit(self, input_data, partial):
        """General fitting method for input data.

        Parameters
        ----------
        input_data : dict of np.ndarray
            dict mapping input names to input data arrays.
        partial : bool
            whether this is a partial or full fit.
        """
        self._reload()
        self._load_inputs(input_data)
        for index, node in enumerate(self.path):
            if utils.generic.isinstance_str(node, 'TransformationNode'):
                try:
                    node.partial_fit() if partial else node.fit()
                    node.transform()
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
        self._reload()

    def _transform(self, input_data):
        """Execute all non-cached nodes along the path given input data.

        Can cache the result for a path if requested.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNode TransformationNodes to begin execution.

        Returns
        -------
        np.ndarray
            the data for the target node of the path given the input data.
        """
        self._reload()
        self._load_inputs(input_data)

        # run transformation nodes to end of path
        for index, node in enumerate(self.path):
            if utils.generic.isinstance_str(node, 'TransformationNode'):
                try:
                    node.transform()
                except Exception as e:
                    print("Error thrown at node named {}".format(node.name))
                    raise
            # erase data from nodes once unneeded
            for predecessor in self.path[:index]:
                outbound_run = all(out_node.has_run for out_node in predecessor.outbound_nodes)
                if outbound_run and predecessor not in self.outputs:
                    predecessor.output = None

    def partial_fit(self, input_data):
        """Fit to input data in an incremental way if possible.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        """
        input_data, _ = input_data
        self._fit(input_data, True)

    def fit(self, input_data):
        """Fit to input data and overwrite the metadata.

        Parameters
        ----------
        input_data : 2-tuple of dict of Numpy array, Numpy array
            the input data to be passed to InputNodes to begin execution, and the index.
        """
        input_data, _ = input_data
        self._fit(input_data, False)

    def fit_generator(self, input_generator):
        """Perform partial fit to every batch of input generator.

        Parameters
        ----------
        input_generator : generator of dict of Numpy array
            generator producing input data to be passed to Input nodes.
        """
        for batch in input_generator:
            self.partial_fit(batch)

    def transform(self, input_data, out_type='array'):
        """Execute the graph with some input data, get the output nodes' data.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        out_type : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        """
        if self.eager:
            raise utils.errors.EagerRunError()

        # if data is created manually with Input, it's not likely to come with an index
        if isinstance(input_data, tuple):
            input_data, data_index = input_data
        else:
            nrows = input_data[list(input_data)[0]].shape[0]
            data_index = pd.RangeIndex(stop=nrows)

        self._transform(input_data)
        output_data = {node.name: node.output for node in self.outputs}
        if self.storage:
            self.storage.write(output_data, data_index)
        return utils.pipeline.format_output(output_data, out_type)

    def transform_generator(self, input_generator, out_type='array'):
        """Execute the graph with some input data from a generator, create generator.

        Parameters
        ----------
        input_generator : dict of Numpy array
            generator producing input data to be passed to Input nodes.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        out_type : {'array', 'dataframe'}
            data type to return as. If dataframe, colnames are node names.
        """
        for batch in input_generator:
            yield self.transform(batch, out_type)

    def save(self, save_dir):
        """Store just the nodes without their data (i.e. pre-execution) in a given file.

        Parameters
        ----------
        save_dir : str
            the desired location of the stored nodes, without the filename.
        """
        # store ref to data outside of Pipeline and remove ref to data in TransformationNodes
        data = {}
        for node in self.path:
            data[node] = node.output
            node.output = None
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open('{}/{}{}.pkl'.format(save_dir, self.name, self.version), 'wb') as f:
            # keep same cache_dir too for new pipeline when loaded
            pipeline_info = {'inputs': self.inputs, 'path': self.path,
                             'outputs': self.outputs, 'name': self.name, 'version': self.version}
            if self.storage:
                storage_info = {'output_names': self.storage.output_names,
                                'dtypes': self.storage.dtypes,
                                'original_shapes': self.storage.original_shapes}
                pipeline_info.update(storage_info)
            pickle.dump(pipeline_info, f)
        # reinsert data into Pipeline
        for node in self.path:
            node.output = data[node]


def load_pipeline(filepath, storage_db=None):
    """Load a set of nodes from a given file, stored previously with Pipeline.save().

    Parameters
    ----------
    filepath : str
        the file from which to load a Pipeline.
    storage_db : Connection (default: sqlite3.connect('megatron_default.db'))
        database connection object to query for cached data from the pipeline.
    """
    with open(filepath, 'rb') as f:
        stored = pickle.load(f)
    P = Pipeline(stored['inputs'], stored['outputs'], stored['name'],
                 stored['version'], storage_db)
    if storage_db:
        # storage members that were calculated during writing
        P.storage.output_names = stored['output_names']
        P.storage.dtypes = stored['dtypes']
        P.storage.original_shapes = stored['original_shapes']
    P.path = stored['path']
    return P
