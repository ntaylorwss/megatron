import numpy as np
from ..nodes.auxiliary import MetricNode
from .. import utils


class Metric:
    """Layer type that holds an evaluation metric; only incorporated for Pipeline evaluation.

    Parameters
    ----------
    metric_fn : function
        the metric function to be wrapped.
    **kwargs
        any keyword arguments to be passed to the metric when being called.
    """
    def __init__(self, metric_fn, **kwargs):
        self.metric = metric_fn
        self.kwargs = kwargs

    def _call(self, nodes, name):
        """Produce a MetricNode acting on the given inbound nodes.

        Parameters
        ----------
        nodes : list of megatron.Node
            nodes to be passed to the metric function.
        name : str
            name to be associated with this MetricNode.
        """
        nodes = utils.generic.listify(nodes)
        out_node = MetricNode(self, nodes, name)
        for node in nodes:
            node.outbound_nodes.append(out_node)
        if all(node.output is not None for node in nodes):
            out_node.evaluate()
        return out_node

    def __call__(self, nodes, name):
        return self._call(nodes, name)

    def evaluate(self, *inputs):
        """Run metric function on given input data."""
        return self.metric(*inputs, **self.kwargs)
