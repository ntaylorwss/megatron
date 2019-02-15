from .core import Node, TransformationNode


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


class KerasNode(TransformationNode):
    """A particular TransformationNode that holds a Keras Layer."""
    def partial_fit(self):
        inputs = [node.output for node in self.inbound_nodes]
        self.layer.fit(*inputs)
        self._clear_inbounds()

    def fit(self, epochs=1):
        inputs = [node.output for node in self.inbound_nodes]
        self.layer.fit(*inputs, epochs=epochs)
        self._clear_inbounds()

    def fit_generator(self, generator, steps_per_epoch, epochs=1):
        """Execute Keras model's fit_generator method.

        Parameters
        ----------
        generator : generator
            data generator to be fit to. Should yield tuples of (observations, labels).
        steps_per_epoch : int
            number of batches that are considered one full epoch.
        epochs : int
            number of epochs to run for.
        """
        self.layer.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
