import unittest
import numpy as np
from sklearn.metrics import f1_score
from megatron.layers.metrics import Metric


class test_Metric(unittest.TestCase):
    def setUp(self):
        self.metric = Metric(f1_score)
        self.X = np.ones(100)
        self.Y = np.ones(100)

    def test_evaluate(self):
        self.assertAlmostEqual(self.metric.evaluate(self.X, self.Y), 1.0)

    def test_call(self):
        nodes = [0, 0]
        new_node = self.metric(nodes)
        assert len(new_node.inbound_nodes) == 2
