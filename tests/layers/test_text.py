import unittest
import numpy as np
from megatron.layers.text import RemoveStopwords


class test_RemoveStopwords(unittest.TestCase):
    def test_transform(self):
        X = 'the it is a the a is it'
        assert RemoveStopwords().transform(X) == ''
        X = 'the it is a the a is it garbage'
        assert RemoveStopwords().transform(X) == 'garbage'
