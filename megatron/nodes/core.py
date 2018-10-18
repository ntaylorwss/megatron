from .. import utils


class Node:
    """Base class of pipeline nodes.

    Parameters
    ----------
    inbound_nodes : list of megatron.Node
        nodes who are to be connected as inputs to this node.
    name : str
        name to give the node.

    Attributes
    ----------
    inbound_nodes : list of megatron.Node
        nodes who are to be connected as inputs to this node.
    name : str
        name to give the node.
    outbound_nodes : list of megatron.Node
        nodes to whom this node is connected as an input.
    output : np.ndarray
        holds the data output by the node's having been run on its inputs.
    has_run : bool
        indicates whether the node has executed a transformation.
    """
    def __init__(self, inbound_nodes, name):
        self.inbound_nodes = inbound_nodes
        self.name = name
        self.outbound_nodes = []
        self.output = None
        self.has_run = False


class InputNode(Node):
    """A pipeline node holding input data as a Numpy array.

    It is always an initial node in a Pipeline (has no input nodes) and, when run,
    stores its given data (either from a feed dict or a function call) in its output.

    Parameters
    ----------
    name : str
        a name to associate with the data; the keys of the Pipeline feed dict will be these names.
    shape : tuple of int
        the shape, not including the observation dimension (1st), of the Numpy arrays to be input.

    Attributes
    ----------
    name : str
        a name to associate with the data; the keys of the Pipeline feed dict will be these names.
    shape : tuple of int
        the shape, not including the observation dimension (1st), of the Numpy arrays to be input.
    output : np.ndarray
        is None until node is run; when run, the Numpy array passed in is stored here.
    """
    def __init__(self, name, shape=()):
        self.is_default_name = False
        self.layer_name = 'Input'
        self.shape = shape
        super().__init__([], name)

    def load(self, observations):
        """Validate and store the data passed in.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        self.validate_input(observations)
        self.output = observations
        self.has_run = True

    def validate_input(self, observations):
        """Ensure shape of data passed in aligns with shape of the node.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        if hasattr(observations, 'shape') and (list(observations.shape[1:]) != list(self.shape)):
            raise utils.errors.ShapeError(self.name, self.shape, observations.shape[1:])

    def __call__(self, observations):
        """Run the node, storing the given data. Intended for eager execution.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        self.transform(observations)
        return self


class TransformationNode(Node):
    """A pipeline node holding a Transformation.

    It connects to a set of input Nodes (of class Node or Input) and, when run,
    applies its given Transformation, storing the result in its output variable.

    Parameters
    ----------
    transformation : megatron.Transformation
        the transformation to be applied to the data from its input Nodes.
    inbound_nodes : list of megatron.Node / megatron.Input
        the Nodes to be connected to this node as input.

    Attributes
    ----------
    transformation : megatron.Transformation
        the transformation to be applied to the data from its input Nodes.
    output : None or np.ndarray
        is None until Node is run; when run, the Numpy array produced is stored here.
    is_fitted : bool
        indicates whether the Transformation inside the Node
        has, if necessary, been fit to data.
    """
    def __init__(self, layer, inbound_nodes, name=None):
        self.is_default_name = name is None
        self.layer = layer
        self.layer_name = layer.name
        super().__init__(inbound_nodes, name)

    @property
    def metadata(self):
        return self.layer.metadata

    @staticmethod
    def _get_outputs(nodes):
        # if one of the arguments to the layer is a list of nodes, dig in to grab the data
        return [node.output for node in nodes]

    def partial_fit(self):
        inputs = self._get_outputs(self.inbound_nodes)
        self.layer.partial_fit(*inputs)
        self.has_run = True

    def fit(self):
        """Calculates metadata based on provided data."""
        inputs = self._get_outputs(self.inbound_nodes)
        self.layer.fit(*inputs)
        self.has_run = True

    def transform(self):
        """Stores result of given Transformation on input Nodes in output variable."""
        inputs = self._get_outputs(self.inbound_nodes)
        self.output = self.layer.transform(*inputs)
        self.has_run = True
