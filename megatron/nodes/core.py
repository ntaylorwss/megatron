from .. import utils


class Node:
    """Base class of pipeline nodes.

    Parameters
    ----------
    inbound_nodes : list of megatron.Node
        nodes who are to be connected as inputs to this node.

    Attributes
    ----------
    inbound_nodes : list of megatron.Node
        nodes who are to be connected as inputs to this node.
    outbound_nodes : list of megatron.Node
        nodes to whom this node is connected as an input.
    output : np.ndarray
        holds the data output by the node's having been run on its inputs.
    outbounds_run : int
        number of outbound nodes that have been executed.
        this is a helper for efficiently removing unneeded data.
    """
    def __init__(self, inbound_nodes):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.output = None
        self.outbounds_run = 0
        self.is_output = False
        self.is_eager = False

    def traverse(self, *path):
        """Return a Node from elsewhere in the graph by navigating to it from this Node.

        A negative number indicates moving up to a parent, a positive number down to a child.
        The number itself is a 1-based index into the parents/children, from left to right.
        For example, a step of -2 will go to the second parent, while a step of 3 will go to
        the third child.

        Parameters
        ----------
        path : *ints
            Arbitrary number of integers indicating the steps in the path.

        Returns
        -------
        Node
            the node at the end of the provided path.
        """
        node = self
        for step in path:
            if step < 0:
                node = node.inbound_nodes[-step-1]
            elif step > 0:
                node = node.outbound_nodes[step-1]
        return node


class InputNode(Node):
    """A pipeline node holding input data as a Numpy array.

    It is always an initial node in a Pipeline (has no inbound nodes) and, when run,
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
    """
    def __init__(self, name, shape=()):
        self.name = name
        self.shape = shape
        super().__init__([])

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
        self.load(observations)
        self.is_eager = True
        return self


class TransformationNode(Node):
    """A pipeline node holding a Transformation.

    It connects to a set of input Nodes (of class Node or Input) and, when run,
    applies its given Transformation, storing the result in its output variable.

    Parameters
    ----------
    layer : megatron.Layer
        the Layer to be applied to the data from its inbound Nodes.
    inbound_nodes : list of megatron.Node / megatron.Input
        the Nodes to be connected to this node as input.
    layer_out_index : int (default: 0)
        when a Layer has multiple return values, shows which one corresponds to this node.

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
    def __init__(self, layer, inbound_nodes, layer_out_index=0):
        super().__init__(inbound_nodes)
        self.layer = layer
        self.layer_out_index = layer_out_index
        self.num_path_outbounds = None

    def _clear_inbounds(self):
        for in_node in self.inbound_nodes:
            if (not in_node.is_output) and in_node.outbounds_run == len(in_node.outbound_nodes):
                in_node.output = None

    def partial_fit(self):
        """Apply partial fit method from Layer to inbound Nodes' data."""
        inputs = [node.output for node in self.inbound_nodes]
        self.layer.partial_fit(*inputs)

    def fit(self):
        """Apply fit method from Layer to inbound Nodes' data."""
        inputs = [node.output for node in self.inbound_nodes]
        try:
            self.layer.fit(*inputs)
        except Exception:
            print("Error thrown by layer named {}".format(self.layer.name))
            raise

    def transform(self, prune=True):
        """Apply and store result of transform method from Layer on inbound Nodes' data.

        Parameters
        ----------
        prune : bool (default: True)
            whether to erase data from intermediate nodes after they are fully used.
        """
        inputs = [node.output for node in self.inbound_nodes]
        try:
            self.output = utils.generic.listify(self.layer.transform(*inputs))[self.layer_out_index]
        except Exception:
            print("Error thrown by layer named {}".format(self.layer.name))
            raise

        if any(in_node.is_eager for in_node in self.inbound_nodes):
            self.is_eager = True
        elif prune:
            for in_node in self.inbound_nodes:
                in_node.outbounds_run += 1
            self._clear_inbounds()
