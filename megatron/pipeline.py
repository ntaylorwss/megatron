import os
import inspect
import sqlite3
import numpy as np
import pandas as pd
import dill as pickle
from collections import defaultdict
from . import utils
from . import io
from .nodes.core import InputNode, TransformationNode
from .nodes.auxiliary import MetricNode, ExploreNode, KerasNode
from .layertools import wrappers


class Pipeline:
    """Holds the core computation graph that maps out Layers and manipulates data.

    Parameters
    ----------
    inputs : list of megatron.Node(s)
        input nodes of the Pipeline, where raw data is fed in.
    outputs : list of megatron.Node(s)
        output nodes of the Pipeline, the processed features.
    name : str
        unique identifying name of the Pipeline.
    version : str
        version tag for Pipeline's cache table in the database.
    storage_db : Connection (defeault: 'sqlite')
        database connection to be used for input and output data storage.

    Attributes
    ----------
    inputs : list of megatron.Node(s)
        input nodes of the Pipeline, where raw data is fed in.
    outputs : list of megatron.Node(s)
        output nodes of the Pipeline, the processed features.
    path : list of megatron.Nodes
        full topological sort of Pipeline from inputs to outputs.
    nodes : dict of list of megatron.Node(s)
        separation of Nodes by type.
    eager : bool
        when True, TransformationNode outputs are to be calculated on creation. This is indicated by
        data being passed to an InputNode node as a function call.
    name : str
        unique identifying name of the Pipeline.
    version : str
        version tag for Pipeline's cache table in the database.
    storage: Connection (defeault: None)
        storage database for input and output data.
    """
    def __init__(self, inputs, outputs, metrics=[], explorers=[],
                 name=None, version=None, storage=None, overwrite=False):
        self.eager = False
        self.inputs = utils.flatten(utils.listify(inputs))
        self.outputs = utils.flatten(utils.listify(outputs))
        self.metrics = utils.flatten(utils.listify(metrics))
        self.explorers = utils.flatten(utils.listify(explorers))

        self.path = utils.pipeline.topsort(self.outputs)
        self.metric_path = utils.pipeline.topsort(self.metrics)
        self.explore_path = utils.pipeline.topsort(self.explorers)
        self.nodes = list(set(self.path + self.metric_path + self.explore_path))

        # wipe output data and learned metadata in case of eager execution
        for node in self.nodes:
            node.output = None
            if hasattr(node, 'metadata'):
                node.metadata = {}

        # identify output nodes
        for node in self.outputs:
            node.is_output = True

        # remove edges that aren't in path
        for node in self.nodes:
            node.inbound_nodes = [in_node for in_node in node.inbound_nodes
                                  if in_node in self.nodes]
            node.outbound_nodes = [out_node for out_node in node.outbound_nodes
                                   if out_node in self.nodes]

        # ensure input data matches with input nodes
        missing_inputs = (set(self.nodes).intersection(self.inputs) - set(self.inputs))
        if len(missing_inputs) > 0:
            raise utils.errors.DisconnectedError(missing_inputs)
        extra_inputs = set(self.nodes).intersection(self.inputs) - set(self.nodes)
        if len(extra_inputs) > 0:
            utils.errors.ExtraInputsWarning(extra_inputs)

        # setup output data storage
        self.name = name
        self.version = version
        if self.version:
            version = str(self.version).replace('.', '_')
        if storage:
            self.storage = io.storage.DataStore(self.name, version, storage, overwrite)
        else:
            self.storage = None

    def _reset_run(self, nodes=None):
        nodes = nodes if nodes else self.nodes
        for node in nodes:
            node.outbounds_run = 0

    def _load_inputs(self, input_data, nodes=None):
        # load data into its corresponding InputNodes
        if nodes is None:
            nodes = self.inputs
        for node in nodes:
            node.load(input_data[node.name])

    def _fit_generator_node(self, node, input_generator, steps_per_epoch, epochs):
        # fit a single node that is not a Keras model to a generator
        path_nodes = utils.pipeline.topsort(node)[:-1]
        input_nodes = [node for node in path_nodes if isinstance(node, InputNode)]
        transform_nodes = [node for node in path_nodes if isinstance(node, TransformationNode)]
        for i, batch in enumerate(input_generator):
            self._load_inputs(batch, input_nodes)
            for parent_node in transform_nodes:
                parent_node.transform(prune=False)
            node.partial_fit()
            if i == (steps_per_epoch * epochs): break

    def _fit_generator_keras(self, node, input_generator, steps_per_epoch, epochs):
        # fit a single node that is a Keras model to a generator
        def _generator(node, input_generator):
            path_nodes = utils.pipeline.topsort(node)[:-1]
            input_nodes = [node for node in path_nodes if isinstance(node, InputNode)]
            transform_nodes = [node for node in path_nodes if isinstance(node, TransformationNode)]
            out_nodes = node.inbound_nodes
            while True:
                for batch in input_generator:
                    self._load_inputs(batch, input_nodes)
                    for node in transform_nodes:
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
        self._load_inputs(input_data)
        for node in self.path:
            if not isinstance(node, TransformationNode): continue
            node.partial_fit()
            node.transform()
        self._reset_run()

    def fit(self, input_data, epochs=1):
        """Fit to input data and overwrite the metadata.

        Parameters
        ----------
        input_data : 2-tuple of dict of Numpy array, Numpy array
            the input data to be passed to InputNodes to begin execution, and the index.
        epochs : int (default: 1)
            number of passes to perform over the data.
        """
        self._load_inputs(input_data)
        for node in self.path:
            if not isinstance(node, TransformationNode): continue
            if isinstance(node, KerasNode):
                node.fit(epochs=epochs)
            else:
                node.fit()
            node.transform()
        self._reset_run()

    def fit_generator(self, input_generator, steps_per_epoch, epochs=1):
        """Fit to generator of input data batches. Execute partial_fit to each batch.

        Parameters
        ----------
        input_generator : generator of 2-tuple of dict of Numpy array and Numpy array
            generator that produces features and labels for each batch of data.
        steps_per_epoch : int
            number of batches that are considered one full epoch.
        epochs : int (default: 1)
            number of passes to perform over the data.
        """
        if sum(isinstance(node, KerasNode) for node in self.path) > 1:
            raise ValueError("Multiple Keras nodes cannot be present when fitting to generator")
        for node in self.path:
            if not isinstance(node, TransformationNode): continue
            if isinstance(node, KerasNode):
                self._fit_generator_keras(node, input_generator, steps_per_epoch, epochs)
            else:
                self._fit_generator_node(node, input_generator, steps_per_epoch, epochs)

    def transform(self, input_data, index_field=None, prune=True):
        """Execute the graph with some input data, get the output nodes' data.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        index_field : str
            name of key from input_data to be used as index for storage and lookup.
        keep_data : bool
            whether to keep data in non-output nodes after execution.
            activating this flag can be useful for debugging.
        """
        if index_field:
            index = input_data.pop(index_field)
            if len(index.shape) > 1:
                raise ValueError("Index field cannot be multi-dimensional array; must be 1D")
        else:
            nrows = input_data[list(input_data)[0]].shape[0]
            index = pd.RangeIndex(stop=nrows)

        self._load_inputs(input_data)

        # run transformation nodes to end of path
        for node in self.path:
            if not isinstance(node, TransformationNode): continue
            node.transform(prune=prune)
        if prune:
            self._reset_run()

        output_data = [node.output for node in self.outputs]
        if self.storage:
            self.storage.write(output_data, index)
        return output_data

    def transform_generator(self, input_generator, steps, index=None):
        """Execute the graph with some input data from a generator, create generator.

        Parameters
        ----------
        input_generator : dict of Numpy array
            generator producing input data to be passed to Input nodes.
        steps : int
            number of batches to pull from input_generator before terminating.
        """
        for i, batch in enumerate(input_generator):
            if i == steps: StopIteration()
            yield self.transform(batch, index, prune=False)
        self._reset_run()

    def evaluate(self, input_data, prune=True):
        """Execute the metric Nodes in the Pipeline and get their results.

        Parameters
        ----------
        input_data : dict of Numpy array
            the input data to be passed to InputNodes to begin execution.
        """
        self._load_inputs(input_data)
        for node in self.metric_path:
            if isinstance(node, TransformationNode):
                node.transform(prune=prune)
            elif isinstance(node, MetricNode):
                node.evaluate()
        self._reset_run()
        return {node.name: node.output
                for node in self.metric_path if isinstance(node, MetricNode)}

    def evaluate_generator(self, input_generator, steps):
        """Execute the metric Nodes in the Pipeline for each batch in a generator."""
        for i, batch in enumerate(input_generator):
            if i == steps: StopIteration()
            yield self.evaluate(batch, prune=False)

    def explore(self, input_data, prune=True):
        self._load_inputs(input_data)
        for node in self.explore_path:
            if isinstance(node, TransformationNode):
                node.transform(prune=prune)
            elif isinstance(node, ExploreNode):
                node.explore()
        self._reset_run()
        return {node.name: node.output
                for node in self.explore_path if isinstance(node, ExploreNode)}

    def explore_generator(self, input_generator, steps):
        """Execute the explorer Nodes in the Pipeline for each batch in a generator."""
        for i, batch in enumerate(input_generator):
            if i == steps: StopIteration()
            yield self.explore(batch, prune=False)

    def save(self, save_dir):
        """Store the Pipeline and its learned metadata without the outputs on disk.

        The filename will be {name of the pipeline}{version}.pkl.

        Parameters
        ----------
        save_dir : str
            the desired location of the stored nodes, without the filename.
        """
        # store ref to data outside of Pipeline and remove ref to data in TransformationNodes
        data = {}
        for node in self.nodes:
            data[node] = node.output
            node.output = None
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # write to disk
        with open('{}/{}{}.pkl'.format(save_dir, self.name, self.version), 'wb') as f:
            # keep same cache_dir too for new pipeline when loaded
            pipeline_info = {'inputs': self.inputs, 'outputs': self.outputs,
                             'path': self.path, 'metric_path': self.metric_path,
                             'explore_path': self.explore_path,
                             'metrics': self.metrics, 'explorers': self.explorers,
                             'name': self.name, 'version': self.version}
            if self.storage:
                storage_info = {'output_names': self.storage.output_names,
                                'dtypes': self.storage.dtypes,
                                'original_shapes': self.storage.original_shapes}
                pipeline_info.update(storage_info)
            pickle.dump(pipeline_info, f)
        # reinsert data into Pipeline
        for node in self.nodes:
            node.output = data[node]


def load_pipeline(filepath, storage_db=None):
    """Load a set of nodes from a given file, stored previously with Pipeline.save().

    Parameters
    ----------
    filepath : str
        the file from which to load a Pipeline.
    storage_db : Connection (default: sqlite3.connect('megatron_default.db'))
        database connection object to query for cached data from the Pipeline.
    """
    with open(filepath, 'rb') as f:
        stored = pickle.load(f)
    P = Pipeline(stored['inputs'], stored['outputs'], stored['metrics'], stored['explorers'],
                 stored['name'], stored['version'], storage_db)
    if storage_db:
        # storage members that were calculated during writing
        P.storage.output_names = stored['output_names']
        P.storage.dtypes = stored['dtypes']
        P.storage.original_shapes = stored['original_shapes']
    return P
