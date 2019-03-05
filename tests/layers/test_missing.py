import unittest
import numpy as np
from megatron.layers.missing import Impute


class test_Impute(unittest.TestCase):
    def test_init(self):
        # empty dict (not allowed)
        self.assertRaises(ValueError, Impute, {})
        # non-dict argument
        self.assertRaises(TypeError, Impute, [1,2,3])
        # value is iterable (not allowed)
        self.assertRaises(TypeError, Impute, {0: [1,2,3]})
        # normal dict works
        Impute({float('nan'): 0})

    def test_transform(self):
        # key that doesn't appear in array (does nothing)
        X = np.ones(50)
        output = Impute({2: 0}).transform(X)
        correct_output = np.ones(50)
        assert np.array_equal(output, correct_output)

        # proper scenario
        X = np.ones(50)
        for i in [2, 6, 10, 14]:
            X[i] = np.nan
        output = Impute({np.nan: 1}).transform(X)
        correct_output = np.ones(50)
        assert np.array_equal(output, correct_output)
