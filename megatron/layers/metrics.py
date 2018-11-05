import numpy as np
from ..nodes.core import MetricNode


class Metric:
    """Base class of metrics."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, nodes, name):
        """Produce a MetricNode acting on the given inbound nodes.

        Parameters
        ----------
        nodes : list of megatron.Node
            nodes to be passed to the metric function.
        name : str
            name to be associated with this MetricNode.
        """
        out_node = MetricNode(self, nodes, name)
        for node in nodes:
            node.outbound_nodes.append(out_node)
        if all(node.output is not None for node in nodes):
            out_node.evaluate()
        return out_node

    def evaluate(self, *inputs):
        """Run metric function on given input data."""
        raise NotImplementedError


class SklearnMetric(Metric):
    """Wrapper for Sklearn metric function.

    Parameters
    ----------
    sklearn_metric : sklearn.Metric
        the metric function to be wrapped.
    **kwargs
        any keyword arguments to be passed to the metric when being called.
    """
    def __init__(self, sklearn_metric, **kwargs):
        self.metric = sklearn.metric
        self.kwargs = kwargs

    def evaluate(self, *inputs):
        self.metric(*inputs, **self.kwargs)


class Accuracy(Metric):
    """Calculates classification accuracy for discrete labels and predictions."""
    def evaluate(self, y_true, y_pred):
        return (y_true == y_pred).mean()
