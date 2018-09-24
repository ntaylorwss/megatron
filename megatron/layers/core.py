import inspect
from ..nodes import InputNode, TransformationNode
from .. import utils


class Layer:
    """Base class of all layers."""
    def _call_on_nodes(self, nodes):
        """Create new node by applying transformation to given nodes.

        Parameters
        ----------
        nodes : megatron.Node(s)
            nodes given as input to the layer's transformation.
        """
        nodes = utils.listify(nodes)
        out_node = TransformationNode(self, nodes)
        for node in nodes:
            node.output_nodes.append(out_node)
        if out_node.pipeline.eager:
            out_node.run()
        return out_node

    def _call_on_feature_set(self, feature_set):
        """Create one new TransformationNode for each node in given feature set.

        Parameters
        ----------
        feature_set : megatron.FeatureSet
            feature set to map the transformation onto.
        """
        new_nodes = [self._call_on_nodes(node) for node in feature_set.nodes]
        return FeatureSet(new_nodes, feature_set.names)

    def __call__(self, inbound_nodes):
        """Creates a TransformationNode associated with this Transformation and the given InputNodes.

        Parameters
        ----------
        inbound_nodes : list of megatron.InputNode / megatron.TransformationNode or megatron.FeatureSet
            the input nodes, whose data are to be passed to transform_fn when run.
        """
        if isinstance(inbound_nodes[0], FeatureSet):
            return self._call_on_feature_set(inbound_nodes[0])
        else:
            return self._call_on_nodes(inbound_nodes)

    def fit(self, *inputs):
        """Calculates and overwrites metadata (if necessary).cored on current inputs.

        Parameters
        ----------
        inputs : numpy.ndarray(s)
            the input data to be fit to; could be one array or a list of arrays.
        """
        pass

    def transform(self, *inputs):
        """Apply transformation to given input data.

        Parameters
        ----------
        inputs : np.ndarray(s)
            input data to be transformed; could be one array or a list of arrays.
        """
        raise NotImplementedError


class StatelessLayer(Layer):
    """A layer holding a stateless transformation.

    Parameters
    ----------
    name : str (optional)
        name to give the layer, used in visualization.
    **kwargs : dict
        the hyperparameters of the transformation function.

    Attributes
    ----------
    name : str (optional)
        name of the layer, used in visualization.
    kwargs : dict
        the hyperparameters of the transformation function.
    """
    def __init__(self, name=None, **kwargs):
        self.name = name if name else self.__class__.__name__
        self.kwargs = kwargs

    def __str__(self):
        """Used in caching pipelines."""
        out = [str(hp) for hp in self.kwargs.values()]
        out.append(inspect.getsource(self.transform))
        return ''.join(out)


class StatefulLayer(Layer):
    """A layer holding a stateful transformation.

    For custom functions that are stateful, and thus require to be fit,
    writing a Transformation subclass is required rather than using a Lambda wrapper.

    Parameters
    ----------
    name : str (optional)
        name of the layer, used in visualization.
    **kwargs : dict
        the hyperparameters of the transformation function.

    Attributes
    ----------
    name : str (optional)
        name of the layer, used in visualization.
    kwargs : dict
        the hyperparameters of the transformation function.
    metadata : dict
        stores any necessary metadata, which is defined by child class.
    """
    def __init__(self, name=None, **kwargs):
        self.name = name if name else self.__class__.__name__
        self.kwargs = kwargs
        self.metadata = {}

    def __str__(self):
        """Used in caching pipelines."""
        metadata = ''.join([utils.md5_hash(metadata) for metadata in self.metadata.values()])
        kwargs = [str(hp) for hp in self.kwargs.values()]
        return '{}{}'.format(inspect.getsource(self.transform), metadata, kwargs)

    def fit(self, *inputs):
        """Must be implemented for a layer to be considered stateful."""
        raise NotImplementedError
