import inspect
from ..nodes import InputNode, TransformationNode
from ..nodes.wrappers import FeatureSet
from .. import utils


class Layer:
    """Base class of all layers.

    Parameters
    ----------
    **kwargs
        hyperparameters of the transformation function.

    Attributes
    ----------
    name : str
        name for the layer, which is the class name.
    kwargs
        hyperparameters of the transformation function.
    """
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs

    def _call_on_nodes(self, nodes, out_name):
        """Create new node by applying transformation to given nodes.

        Parameters
        ----------
        nodes : megatron.Node(s)
            nodes given as input to the layer's transformation.
        out_name : str
            name to give the newly created node.
        """
        out_node = TransformationNode(self, nodes, out_name)
        for node in nodes:
            node.outbound_nodes.append(out_node)
        if all(node.output is not None for node in nodes):
            out_node.fit()
            out_node.transform()
        return out_node

    def _call_on_feature_set(self, feature_set):
        """Create one new TransformationNode for each node in given feature set.

        Parameters
        ----------
        feature_set : megatron.FeatureSet
            feature set to map the transformation onto.
        """
        new_nodes = [self._call_on_nodes([node], node.name) for node in feature_set.nodes]
        return FeatureSet(new_nodes)

    def __call__(self, inbound_nodes, name=None):
        """Creates a TransformationNode associated with this Transformation and the given InputNodes.

        When running eagerly, perform a fit and transform, and store the result of the transform in output member.

        Parameters
        ----------
        inbound_nodes : list of megatron.InputNode / megatron.TransformationNode or megatron.FeatureSet
            the input nodes, whose data are to be passed to transform_fn when run.
        name : str (default: None)
            name for the newly created node. Not used for FeatureSet input, and only required for output nodes.
        """
        inbound_nodes = utils.generic.listify(inbound_nodes)
        if isinstance(inbound_nodes[0], FeatureSet):
            if name:
                raise ValueError("When transforming FeatureSet, name cannot be provided")
            return self._call_on_feature_set(inbound_nodes[0])
        else:
            return self._call_on_nodes(inbound_nodes, name)

    def partial_fit(self, *inputs):
        """Updates metadata based on given batch of data or full dataset.

        Contains the main logic of fitting.
        Parameters
        ----------
        inputs : numpy.ndarray(s)
            the input data to be fit to; could be one array or a list of arrays.
        """
        pass

    def fit(self, *inputs):
        """Overwrites metadata based on given batch of data or full dataset.

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
    """A layer holding a stateless transformation."""
    pass


class StatefulLayer(Layer):
    """A layer holding a stateful transformation.

    For custom functions that are stateful, and thus require to be fit,
    writing a Transformation subclass is required rather than using a Lambda wrapper.

    Parameters
    ----------
    **kwargs : dict
        the hyperparameters of the transformation function.

    Attributes
    ----------
    name : str (optional)
        name of the layer, which is the class name.
    kwargs : dict
        the hyperparameters of the transformation function.
    metadata : dict
        stores any necessary metadata, which is defined by child class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {}

    def partial_fit(self, *inputs):
        """Updates metadata based on given batch of data or full dataset.

        Contains the main logic of fitting. This is what should be overwritten by all child classes.

        Parameters
        ----------
        inputs : numpy.ndarray(s)
            the input data to be fit to; could be one array or a list of arrays.
        """
        raise NotImplementedError

    def fit(self, *inputs):
        self.metadata = {}
        self.partial_fit(*inputs)


class Lambda(StatelessLayer):
    """A layer holding a stateless transformation.

    For custom functions that are stateless, and thus do not require to be fit,
    a Lambda wrapper is preferred to creating a Transformation subclass.

    Parameters
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    **kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.

    Attributes
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.
    """
    def __init__(self, transform_fn, **kwargs):
        self.transform_fn = transform_fn
        super().__init__(**kwargs)

    def transform(self, *inputs):
        """Applies transform_fn to given input data.

        Parameters
        ----------
        inputs : np.ndarray(s)
            input data to be passed to transform_fn; could be one array or a list of arrays.
        """
        return self.transform_fn(*inputs, **self.kwargs)
