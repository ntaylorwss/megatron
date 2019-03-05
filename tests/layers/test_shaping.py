import unittest
import numpy as np
from megatron.layers.shaping import Cast, AddDim, OneHotRange, OneHotLabels, Reshape, Flatten
from megatron.layers.shaping import SplitDict, TimeSeries, Concatenate, Slice, Filter


class test_Cast(unittest.TestCase):
    def test_init(self):
        self.assertRaises(TypeError, Cast, 0)
        assert Cast(int).kwargs['new_type'] == int

    def test_transform(self):
        # incompatible cast (whole input)
        self.assertRaises(ValueError, Cast(int).transform, np.array(['a', 'b', 'c']))
        # incompatible cast (one element of input)
        self.assertRaises(ValueError, Cast(int).transform, np.array([1., 2., 'c']))
        # valid cast
        assert all(isinstance(v, np.integer) for v in Cast(int).transform(np.array(['1', '2', '3'])))


class test_AddDim(unittest.TestCase):
    def test_transform(self):
        # axis out of range
        self.assertRaises(ValueError, AddDim(axis=2).transform, np.ones(5))
        # axis at front
        assert AddDim(axis=0).transform(np.ones(5)).shape == (1, 5)
        # axis at back
        assert AddDim().transform(np.ones(5)).shape == (5, 1)
        # axis in middle
        assert AddDim(axis=1).transform(np.ones((5, 5))).shape == (5, 1, 5)


class test_OneHotRange(unittest.TestCase):
    def setUp(self):
        self.transformer = OneHotRange()
        self.transformer_strict = OneHotRange(strict=True)

    def test_partial_fit(self):
        self.transformer.partial_fit(np.array([-2, 4]))
        assert self.transformer.metadata['min_val'] == -2 and self.transformer.metadata['max_val'] == 4
        self.transformer.partial_fit(np.array([-3, 5, 4]))
        assert self.transformer.metadata['min_val'] == -3 and self.transformer.metadata['max_val'] == 5
        self.transformer.partial_fit(np.array([[1, 2], [1, 2]]))
        assert self.transformer.metadata['min_val'] == -3 and self.transformer.metadata['max_val'] == 5

    def test_fit(self):
        self.transformer.fit(np.array([-2, 4]))
        assert self.transformer.metadata['min_val'] == -2 and self.transformer.metadata['max_val'] == 4
        self.transformer.fit(np.array([[1, 2], [1, 2]]))
        assert self.transformer.metadata['min_val'] == 1 and self.transformer.metadata['max_val'] == 2

    def test_transform(self):
        self.assertRaises(TypeError, OneHotRange().fit, np.array(['a', 'b', 'c']))
        self.assertRaises(TypeError, OneHotRange().fit, np.array([1.5, 2, 3]))

        self.transformer_strict = OneHotRange(strict=True)
        self.transformer = OneHotRange()

        # value out of range (strict)
        self.transformer_strict.fit(np.array([-2, 1, 2, 3]))
        self.assertRaises(ValueError, self.transformer_strict.transform, np.array([-2, 0, 2, 4]))

        # value out of range (non-strict)
        self.transformer.fit(np.array([-2, 1, 2, 3]))
        assert np.array_equal(self.transformer.transform(np.array([-2, 0, 2, 4]))[-1], np.zeros(6))

        # 1D full range
        self.transformer.fit(np.array([1., 2, 3]))
        output = self.transformer.transform(np.array([1., 2, 3]))
        correct_output = np.zeros((3, 3))
        for i, j in enumerate([0, 1, 2]):
            correct_output[i, j] = 1
        assert np.array_equal(output, correct_output)

        # 1D sparse
        self.transformer.fit(np.array([-1, 2, 5, 7]))
        output = self.transformer.transform(np.array([5, 2, -1, 7]))
        correct_output = np.zeros((4, 9))
        for i, j in enumerate([6, 3, 0, 8]):
            correct_output[i, j] = 1
        assert np.array_equal(output, correct_output)

        # 2D
        self.transformer.fit(np.random.randint(-2, 3, (50, 50)))
        assert self.transformer.transform(np.random.randint(-2, 3, (40, 40))).shape == (40, 40, 5)


class test_OneHotLabels(unittest.TestCase):
    def setUp(self):
        self.transformer = OneHotLabels()
        self.transformer_strict = OneHotLabels(strict=True)

    def test_partial_fit(self):
        self.transformer.partial_fit(np.array(['a', 2]))
        assert set(self.transformer.metadata['categories']) == {'a', '2'}
        self.transformer.partial_fit(np.array(['a', 'b', 3]))
        assert set(self.transformer.metadata['categories']) == {'a', '2', 'b', '3'}
        self.transformer.partial_fit(np.array([['a', 'b'], [4, 2]]))
        assert set(self.transformer.metadata['categories']) == {'a', '2', 'b', '3', '4'}

    def test_fit(self):
        self.transformer.fit(np.array(['a', 2]))
        assert set(self.transformer.metadata['categories']) == {'a', '2'}
        self.transformer.fit(np.array([['a', 'b'], [3, 'a']]))
        assert set(self.transformer.metadata['categories']) == {'a', 'b', '3'}

    def test_transform(self):
        self.transformer_strict = OneHotLabels(strict=True)
        self.transformer = OneHotLabels()

        # value out of set (strict)
        self.transformer_strict.fit(['a', 1, 0.5, 'b'])
        self.assertRaises(ValueError, self.transformer_strict.transform, np.array(['a', '1', '0.5', 'b', 'c']))

        # value out of set (non-strict)
        self.transformer.fit(np.array(['a', 1, 0.5, 'c']))
        assert np.array_equal(self.transformer.transform(np.array(['a', '1', '0.5', 'b']))[-1], np.zeros(4))

        self.transformer.fit(np.array(['a', 'b', 'c']))
        output = self.transformer.transform(np.array(['a', 'b', 'c']))
        correct_output = np.zeros((3, 3))
        for i, j in enumerate([0, 1, 2]):
            correct_output[i, j] = 1
        assert np.array_equal(output, correct_output)

        self.transformer.fit(np.array([['a', 'b'], ['c', 'd']]))
        output = self.transformer.transform(np.array([['a', 'b'], ['c', 'd']]))
        assert output.shape == (2, 2, 4)


class test_Reshape(unittest.TestCase):
    def test_transform(self):
        self.assertRaises(ValueError, Reshape((3,2)).transform, np.ones(4))
        assert Reshape((2, 2)).transform(np.ones(4)).shape == (2, 2)


class test_Flatten(unittest.TestCase):
    def test_transform(self):
        assert Flatten().transform(np.ones((2, 2))).shape == (4,)


class test_SplitDict(unittest.TestCase):
    def test_transform(self):
        transformer = SplitDict(['a', 'b'])
        output = transformer.transform(np.array([{'a': 1, 'b': 2, 'c': 3}, {'a': 2, 'b': 3, 'c': 4}]))
        assert all(np.array_equal(a, b) for a, b in zip(output, [np.array([1, 2]), np.array([2, 3])]))


class test_TimeSeries(unittest.TestCase):
    def test_init(self):
        self.assertRaises(ValueError, TimeSeries, 1)
        assert TimeSeries(window_size=3).kwargs['window_size'] == 3

    def fit(self):
        transformer = TimeSeries(window_size=3)
        X = np.arange(100).reshape((20, 5))
        transformer.fit(X)
        assert transformer.metadata['shape'] == (5,)
        assert transformer.metadata['previous'] == np.zeros((2, 5))

    def test_transform(self):
        transformer = TimeSeries(window_size=3, reverse=True)
        X = np.arange(15).reshape((3, 5))
        transformer.fit(X)
        output = transformer.transform(X)
        correct_output = np.array([
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 2, 3, 4]],
            [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
        ])
        assert np.array_equal(output, correct_output)


class test_Concatenate(unittest.TestCase):
    def test_transform(self):
        X = [np.ones(100), np.ones(100), np.ones(100)]
        assert Concatenate().transform(*X).shape == (100, 3)

        X = [np.ones((100, 5)), np.ones((100, 3)), np.ones((100, 2))]
        assert Concatenate(axis=1).transform(*X).shape == (100, 10)
        self.assertRaises(ValueError, Concatenate(axis=0).transform, X[0], X[1], X[2])


class test_Slice(unittest.TestCase):
    def test_transform(self):
        self.assertRaises(ValueError, Slice(2, 2).transform, np.ones(100))

        output = Slice((2, 5), 3).transform(np.arange(100).reshape((10, 10)))
        correct_output = np.array([23, 33, 43])
        assert np.array_equal(output, correct_output)

        output = Slice(3, (0, 2), (0, 3, 2)).transform(np.arange(64).reshape((4, 4, 4)))
        correct_output = np.array([[48, 50], [52, 54]])
        assert np.array_equal(output, correct_output)


class test_Filter(unittest.TestCase):
    def test_transform(self):
        self.assertRaises(TypeError, Filter().transform, np.ones((3, 5)), np.array(['a', 'b', 'c']))

        output = Filter().transform(np.arange(5), np.array([1, 0, 1, 0, 1]))
        correct_output = np.array([0, 2, 4])
        assert np.array_equal(output, correct_output)

        output = Filter().transform(np.arange(16).reshape((4, 4)), np.array([1, 0, 0, 1]))
        correct_output = np.array([[0, 1, 2, 3], [12, 13, 14, 15]])
        assert np.array_equal(output, correct_output)
