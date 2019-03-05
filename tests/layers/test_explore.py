import unittest
import numpy as np
from megatron.layers.explore import Describe, Correlate


class test_Describe(unittest.TestCase):
    def setUp(self):
        self.X_2D = np.ones((100, 5))
        self.X_1D = np.ones(100)
        self.explorer = Describe()

    def test_explore_2D(self):
        output_2D = self.explorer.explore(self.X_2D)

        # test keys
        assert set(output_2D) == set(['mean','sd', 'min', '25%', '50%', '75%', 'max'])

        # test shape
        output_shapes = [stat.shape for stat in output_2D.values()]
        correct_shapes = [(5,) for _ in output_2D]
        assert [out == correct for out, correct in zip(output_shapes, correct_shapes)]

        # test values
        correct_values = {'mean': np.array([1., 1., 1., 1., 1.]),
                          'sd': np.array([0., 0., 0., 0., 0.]),
                          'min': np.array([1., 1., 1., 1., 1.]),
                          '25%': np.array([1., 1., 1., 1., 1.]),
                          '50%': np.array([1., 1., 1., 1., 1.]),
                          '75%': np.array([1., 1., 1., 1., 1.]),
                          'max': np.array([1., 1., 1., 1., 1.])}
        assert all(np.isclose(output_2D[k], correct_values[k]).all() for k in correct_values)

    def test_explore_1D(self):
        output_1D = self.explorer.explore(self.X_1D)

        # test keys
        assert set(output_1D) == set(['mean','sd', 'min', '25%', '50%', '75%', 'max'])

        # test output data type
        output_type = [type(stat) for stat in output_1D.values()]
        correct_type = [float for _ in output_1D]
        assert [out == correct for out, correct in zip(output_type, correct_type)]

        # test values
        correct_values = {'mean': 1., 'sd': 0., 'min': 1., '25%': 1.,
                          '50%': 1., '75%': 1., 'max': 1.}
        assert all(np.isclose(output_1D[k], correct_values[k]) for k in correct_values)


class test_Correlate(unittest.TestCase):
    def setUp(self):
        self.X_2D = (np.ones((100, 5)).T * np.arange(1, 101)).T
        self.X_1D = np.arange(100)
        self.Y_1D = np.arange(100)
        self.explorer = Correlate()

    def test_explore_1D_wrong(self):
        self.assertRaises(ValueError, self.explorer.explore, self.X_1D)

    def test_explore_2D(self):
        output = self.explorer.explore(self.X_2D)
        correct_output = np.ones((5, 5))
        assert np.isclose(output, correct_output).all()

    def test_explore_1D(self):
        output = self.explorer.explore(self.X_1D, self.Y_1D)
        correct_output = np.ones((2, 2))
        assert np.isclose(output, correct_output).all()
