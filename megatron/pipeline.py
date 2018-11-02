import os
import sqlite3
import numpy as np
import pandas as pd
import dill as pickle
from collections import defaultdict
from . import utils
from . import io
from .nodes.core import InputNode, TransformationNode, KerasNode, MetricNode


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
        self.inputs = utils.flatten(utils.listify(inputs))
        self.outputs = utils.flatten(utils.listify(outputs))
        self.path = utils.pipeline.topsort(self.outputs)
        self.nodes = self._split_path(self.path)
        self.nodes['metric'] = self._get_metric_nodes(self.path)

        # ensure input data matches with input nodes
        missing_inputs = (set(self.path).intersection(self.inputs) - set(self.inputs))
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

        # clear data in case of eager execution prior to model build
        self._reload()

    def _split_path(self, path):
        # split nodes up by type
        nodes = {}
        names_and_types = [('transformation', TransformationNode),
                           ('input', InputNode),
                           ('keras', KerasNode)]
        for node_name, node_type in names_and_types:
            nodes[node_name] = [node for node in self.path if isinstance(node, node_type)]
        return nodes

    def _get_metric_nodes(self, path):
        metrics = set()
        for node in path:
            node_metrics = [out_node for out_node in node.outbound_nodes
                            if isinstance(out_node, MetricNode)]
            metrics.update(node_metrics)
        return list(metrics)

    def _reload(self):
        """Reset all nodes' has_run indicators to False, clear data."""
        for node in self.path:
            node.has_run = False
            node.output = None

    def _load_inputs(self, input_data, nodes=None):
        """Load data into Input nodes.

        Parameters
        ----------
        input_data : dict of np.ndarray
            dict mapping input names to input data arrays.
        """
        if nodes is None:
            nodes = self.inputs
        for node in nodes:
            node.load(input_data[node.name])

    def _fit_generator_node(self, node, input_generator, steps_per_epoch, epochs):
        path_nodes = self._split_path(utils.pipeline.topsort(node)[:-1])
        for i, batch in enumerate(input_generator):
            self._load_inputs(batch, path_nodes['input'])
            for parent_node in path_nodes['transformation']:
                parent_node.transform()
            node.partial_fit()
            if i == (steps_per_epoch * epochs): break

    def _fit_generator_keras(self, node, input_generator, steps_per_epoch, epochs):
        def _generator(node, input_generator):
            path_nodes = self._split_path(utils.pipeline.topsort(node)[:-1])
            out_nodes = node.inbound_nodes
            while True:
                for batch in input_generator:
                    self._load_inputs(batch, path_nodes['input'])
                    for node in path_nodes['transformation']:
                        node.transform()
                    yield [node.output for node in out_nodes]

        node.fit_generator(_generator(node, input_generator),
                           steps_per_epoch=steps_per_epoch, epochs=epochs)

    def partial_fit(self, input_data):
        """Fit to input data in an incremental way if possible.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        """
        self._reload()
        self._load_inputs(input_data)
        for node in self.nodes['transformation']:
            node.partial_fit()
            node.transform()
        # restore has_run, clear data
        self._reload()

    def fit(self, input_data, epochs=1):
        """Fit to input data and overwrite the metadata.

        Parameters
        ----------
        input_data : 2-tuple of dict of Numpy array, Numpy array
            the input data to be passed to InputNodes to begin execution, and the index.
        """
        self._reload()
        self._load_inputs(input_data)
        for node in self.nodes['transformation']:
            node.fit(epochs=epochs) if node in self.nodes['keras'] else node.fit()
            node.transform()
        # restore has_run, clear data
        self._reload()

    def fit_generator(self, input_generator, steps_per_epoch, epochs=1):
        if len(self.nodes['keras']) > 1:
            raise ValueError("Multiple Keras nodes cannot be present when fitting to generator")
        for node in self.nodes['transformation']:
            if node in self.nodes['keras']:
                self._fit_generator_keras(node, input_generator, steps_per_epoch, epochs)
            else:
                self._fit_generator_node(node, input_generator, steps_per_epoch, epochs)
        self._reload()

    def transform(self, input_data, index_field=None, keep_data=False):
        """Execute the graph with some input data, get the output nodes' data.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        """
        if index_field:
            index = input_data.pop(index_field)
            if len(index.shape) > 1:
                raise ValueError("Index field cannot be multi-dimensional array; must be 1D")
        else:
            nrows = input_data[list(input_data)[0]].shape[0]
            index = pd.RangeIndex(stop=nrows)

        self._reload()
        self._load_inputs(input_data)

        # run transformation nodes to end of path
        for node in self.nodes['transformation']:
            node.transform()

        output_data = [node.output for node in self.outputs]
        if self.storage:
            self.storage.write(output_data, index)
        return output_data[0] if len(output_data) == 1 else output_data

    def transform_generator(self, input_generator, steps, index=None):
        """Execute the graph with some input data from a generator, create generator.

        Parameters
        ----------
        input_generator : dict of Numpy array
            generator producing input data to be passed to Input nodes.
        cache_result : bool
            whether to store the resulting Numpy array in the cache.
        """
        for i, batch in enumerate(input_generator):
            if i == steps: StopIteration()
            yield self.transform(batch, out_type, index, keep_data=True)

    def evaluate(self, input_data):
        """Execute the metric nodes in the graph, performing a transform to acquire the data.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        """
        self._reload()
        self._load_inputs(input_data)
        for node in self.nodes['transformation']:
            node.transform()
        for node in self.nodes['metric']:
            node.evaluate()
        return {node.name: node.output for node in self.nodes['metric']}

    def evaluate_generator(self, input_generator, steps):
        for i, batch in enumerate(input_generator):
            if i == steps: StopIteration()
            yield self.evaluate(batch)

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
