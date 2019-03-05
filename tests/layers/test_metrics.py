import unittest
import numpy as np
from sklearn.metrics import f1_score
from megatron.layers.metrics import Metric
from megatron.nodes.core import Node


class test_Metric(unittest.TestCase):
    def setUp(self):
        self.metric = Metric(f1_score)
        self.X = np.ones(100)
        self.Y = np.ones(100)

    def test_evaluate(self):
        self.assertAlmostEqual(self.metric.evaluate(self.X, self.Y), 1.0)

    def test_call(self):
        nodes = [Node([]), Node([])]
        new_node = self.metric(nodes, 'f1_score')
        assert len(new_node.inbound_nodes) == 2
