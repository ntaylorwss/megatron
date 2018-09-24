from .. import utils


class Node:
    def __init__(self, pipeline, name, inbound_nodes):
        self.pipeline = pipeline
        self.pipeline._add_node(self)
        self.name = name
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.output = None

    def run(self, inputs):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class InputNode(Node):
    """A pipeline node holding input data as a Numpy array.

    It is always an initial node in a Pipeline (has no input nodes) and, when run,
    stores its given data (either from a feed dict or a function call) in its output.

    Parameters
    ----------
    pipeline : megatron.Pipeline
        the Pipeline with which the node is associated.
    name : str
        a name to associate with the data; the keys of the Pipeline feed dict will be these names.
    input_shape : tuple of int
        the shape, not including the observation dimension (1st), of the Numpy arrays to be input.

    Attributes
    ----------
    pipeline : megatron.Pipeline
        the Pipeline with which the node is associated.
    name : str
        a name to associate with the data; the keys of the Pipeline feed dict will be these names.
    input_shape : tuple of int
        the shape, not including the observation dimension (1st), of the Numpy arrays to be input.
    str : str
        stores the result of magic method str, so that it doesn't have to be recalculated.
    output : np.ndarray
        is None until node is run; when run, the Numpy array passed in is stored here.
    """
    def __init__(self, pipeline, name, input_shape=()):
        super().__init__(pipeline, name, [])
        self.input_shape = input_shape
        self.str = None
        self.outbound_nodes = []
        self.output = None

    def run(self, observations):
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

    def __str__(self):
        """Used in caching subpipelines."""
        if self.str is None:
            self.str = utils.md5_hash(self.output)
        return self.str

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
        if hasattr(observations, 'shape') and (list(observations.shape[1:]) != list(self.input_shape)):
            raise utils.ShapeError(self.name, self.input_shape, observations.shape[1:])

    def __call__(self, observations):
        """Run the node, and indicate to the associated Pipeline that it is running eagerly.

        Parameters
        ----------
        observations : np.ndarray
            data from either the feed dict or the function call, to be validated.

        Raises
        ------
        megatron.utils.ShapeError
            error indicating that the shape of the data does not match the shape of the node.
        """
        self.run(observations)
        self.pipeline.eager = True
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
    pipeline : megatron.Pipeline
        the Pipeline with which this Node is associated; deduced from input nodes.
    output : None or np.ndarray
        is None until Node is run; when run, the Numpy array produced is stored here.
    is_fitted : bool
        indicates whether the Transformation inside the Node
        has, if necessary, been fit to data.
    """
    def __init__(self, transformation, inbound_nodes):
        pipeline = inbound_nodes[0].pipeline
        super().__init__(pipeline, transformation.name, inbound_nodes)
        self.transformation = transformation
        self.is_fitted = False

    def run(self):
        """Stores result of given Transformation on input Nodes in output variable."""
        inputs = [node.output for node in self.inbound_nodes]
        if not self.is_fitted:
            self.transformation.fit(*inputs)
            self.is_fitted = True
        self.output = self.transformation.transform(*inputs)

    def __str__(self):
        """Used in caching subpipelines."""
        if self.str is None:
            self.str = str(self.transformation)
        return self.str
