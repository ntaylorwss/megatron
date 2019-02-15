import inspect
from ..nodes import InputNode, TransformationNode
from .. import utils


class Input(InputNode):
    """Wrapper for Input nodes to make them appear as a Layer, for consistency."""
    pass


class Layer:
    """Base class of all layers.

    Parameters
    ----------
    n_outputs (default: 1)
        number of distinct data, and thus nodes, output by the layer's transform.
    **kwargs
        hyperparameters of the transformation function.

    Attributes
    ----------
    n_outputs
        number of distinct data, and thus nodes, output by the layer's transform.
    kwargs
        hyperparameters of the transformation function.
    """
    def __init__(self, n_outputs=1, **kwargs):
        self.n_outputs = n_outputs
        self.kwargs = kwargs
        self.name = self.__class__.__name__

    def _call(self, nodes):
        """Creates a TransformationNode associated with this Layer and the given InputNode(s).

        When running eagerly, will perform a fit and transform.

        Parameters
        ----------
        nodes : list of megatron.InputNode / megatron.TransformationNode
            the input nodes, whose data are to be passed to transform_fn when run.
        """
        nodes = utils.generic.listify(nodes)

        if self.n_outputs > 1:
            out_nodes = [TransformationNode(self, nodes, i)
                        for i in range(self.n_outputs)]
            for node in nodes:
                node.outbound_nodes += out_nodes
            if all(node.output is not None for node in nodes):
                for out_node in out_nodes:
                    out_node.fit()
                    out_node.transform()
            out = out_nodes
        else:
            out_node = TransformationNode(self, nodes)
            for node in nodes:
                node.outbound_nodes.append(out_node)
            if all(node.output is not None for node in nodes):
                out_node.fit()
                out_node.transform()
            out = out_node
        return out

    def __call__(self, nodes):
        return self._call(nodes)

    def partial_fit(self, *inputs):
        """Update metadata based on given data in an iterative fashion.

        Parameters
        ----------
        inputs : numpy.ndarray(s)
            the input data to be fit to; could be one array or a list of arrays.
        """
        pass

    def fit(self, *inputs):
        """Overwrite metadata based on given data.

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
    n_outputs (default: 1)
        number of distinct data, and thus nodes, output by the layer's transform.
    **kwargs : dict
        the hyperparameters of the transformation function.

    Attributes
    ----------
    n_outputs (default: 1)
        number of distinct data, and thus nodes, output by the layer's transform.
    kwargs : dict
        the hyperparameters of the transformation function.
    metadata : dict
        stores any necessary metadata, which is defined by child class.
    """
    def __init__(self, n_outputs=1, **kwargs):
        super().__init__(n_outputs=n_outputs, **kwargs)
        self.metadata = {}

    def partial_fit(self, *inputs):
        """Update metadata based on given batch of data or full dataset.

        Contains the main logic of fitting. This is what should be overwritten by all child classes.

        Parameters
        ----------
        inputs : numpy.ndarray(s)
            the input data to be fit to; could be one array or a list of arrays.
        """
        msg = "Layer {} does not support partial_fit() or it has not been defined yet"
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def fit(self, *inputs):
        """Overwrite metadata based on given batch of data or full dataset.

        Parameters
        ----------
        inputs : numpy.ndarray(s)
            the input data to be fit to; could be one array or a list of arrays.
        """
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
    n_outputs (default: 1)
        number of distinct data, and thus nodes, output by the layer's transform.
    **kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.

    Attributes
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    n_outputs (default: 1)
        number of distinct data, and thus nodes, output by the layer's transform.
    kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.
    """
    def __init__(self, transform_fn, n_outputs=1, **kwargs):
        self.transform_fn = transform_fn
        super().__init__(n_outputs=n_outputs, **kwargs)

    def transform(self, *inputs):
        """Apply associated function to given input data.

        Parameters
        ----------
        inputs : np.ndarray(s)
            input data to be passed to transform_fn; could be one array or a list of arrays.
        """
        return self.transform_fn(*inputs, **self.kwargs)
