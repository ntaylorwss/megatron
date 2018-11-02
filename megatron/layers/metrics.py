import numpy as np
from ..nodes.core import MetricNode


class Metric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, nodes, name):
        out_node = MetricNode(self, nodes, name)
        for node in nodes:
            node.outbound_nodes.append(out_node)
        if all(node.output is not None for node in nodes):
            out_node.evaluate()
        return out_node

    def evaluate(self, *inputs):
        raise NotImplementedError


class SklearnMetric(Metric):
    def __init__(self, sklearn_metric, **kwargs):
        self.metric = sklearn.metric
        self.kwargs = kwargs

    def evaluate(self, *inputs):
        self.metric(*inputs, **self.kwargs)


class Accuracy(Metric):
    def evaluate(self, y_true, y_pred):
        return (y_true == y_pred).mean()
