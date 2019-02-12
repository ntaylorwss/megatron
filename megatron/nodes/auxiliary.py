from .core import Node


class MetricNode(Node):
    """A Node that holds an evaluation metric.

    These are not run like other nodes. They do not factor into fit and transform.
    They are only used when the Pipeline's evaluate() method is invoked.

    Parameters
    ----------
    metric : function
        the metric function to be applied to the inbound data.
    inbound_nodes : list of megatron.Node
        the nodes whose data are passed to the metric function.
    name : str
        the name of the node. This is used to refer to the result in the output dict.

    Attributes
    ----------
    metric : function
        the metric function to be applied to the inbound data.
    inbound_nodes : list of megatron.Node
        the nodes whose data are passed to the metric function.
    name : str
        the name of the node. This is used to refer to the result in the output dict.
    """
    def __init__(self, layer, inbound_nodes, name):
        self.layer = layer
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.name = name

    def evaluate(self):
        """Apply metric function to inbound data and store result."""
        inputs = [node.output for node in self.inbound_nodes]
        self.output = self.layer.evaluate(*inputs)


class ExploreNode(Node):
    """A node that holds an exploratory tool, such as a visualization or statistical summary.

    These are not run like other nodes. They do not factor into fit and transform.
    They are only used when the Pipeline's explore() method is invoked.

    Parameters
    ----------
    explorer : function
        the function to be applied to the inbound data that will produce the analysis.
    inbound_nodes : list of megatron.Node
        the nodes whose data are passed to the exploration function.
    name : str
        the name of the node. This is used to refer to the result in the output dict.
    """
    def __init__(self, layer, inbound_nodes, name):
        self.layer = layer
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.name = name

    def explore(self):
        """Apply explorer to inbound data and store result."""
        inputs = [node.output for node in self.inbound_nodes]
        self.output = self.layer.explore(*inputs)
