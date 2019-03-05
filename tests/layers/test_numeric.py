import unittest
import numpy as np
from megatron.layers.numeric import Add, Subtract, ScalarMultiply
from megatron.layers.numeric import ElementWiseMultiply, Divide, StaticDot, Dot, Normalize


class test_Add(unittest.TestCase):
    def test_transform(self):
        # mismatched shapes
        X = np.ones(50)
        Y = np.ones(60)
        self.assertRaises(ValueError, Add().transform, X, Y)
        # mismatched types
        X = np.ones(3)
        Y = np.array(['a', 'b', 'c'])
        self.assertRaises(TypeError, Add().transform, X, Y)
        # proper case
        X = np.ones(50)
        Y = np.ones(50)
        assert np.array_equal(Add().transform(X, Y), (np.ones(50) * 2))
        # proper case >1D
        X = np.ones((5, 5))
        Y = np.ones((5, 5))
        assert np.array_equal(Add().transform(X, Y), (np.ones((5, 5)) * 2))


class test_Subtract(unittest.TestCase):
    def test_transform(self):
        # mismatched shapes
        X = np.ones(10)
        Y = np.ones(5)
        self.assertRaises(ValueError, Subtract().transform, X, Y)
        # mismatched types
        X = np.ones(3)
        Y = np.array(['a','b', 'c'])
        self.assertRaises(TypeError, Subtract().transform, X, Y)
        # proper case
        X = np.ones(10) * 2
        Y = np.ones(10)
        assert np.array_equal(Subtract().transform(X, Y), np.ones(10))
        # proper case >1D
        X = np.ones((5, 5)) * 2
        Y = np.ones((5, 5))
        assert np.array_equal(Subtract().transform(X, Y), np.ones((5, 5)))


class test_ScalarMultiply(unittest.TestCase):
    def test_transform(self):
        # proper case
        X = np.ones(10)
        assert np.array_equal(ScalarMultiply(2).transform(X), np.ones(10) * 2)
        # proper case >1D
        X = np.ones((5, 5))
        assert np.isclose(ScalarMultiply(1.5).transform(X), np.ones((5, 5)) * 1.5).all()


class test_ElementWiseMultiply(unittest.TestCase):
    def test_transform(self):
        # mismatched shapes
        X = np.ones(10)
        Y = np.ones(5)
        self.assertRaises(ValueError, ElementWiseMultiply().transform, X, Y)
        # mismatched types
        X = np.ones(3)
        Y = np.array(['a','b', 'c'])
        self.assertRaises(TypeError, ElementWiseMultiply().transform, X, Y)
        # proper case
        X = np.ones(10) * 2
        Y = np.ones(10) * 3
        assert np.array_equal(ElementWiseMultiply().transform(X, Y), np.ones(10) * 6)
        # proper case >1D
        X = np.ones((5, 5)) * 2
        Y = np.ones((5, 5)) * 3
        assert np.array_equal(ElementWiseMultiply().transform(X, Y), np.ones((5, 5)) * 6)


class test_Divide(unittest.TestCase):
    def test_transform(self):
        # mismatched shapes
        X = np.ones(10)
        Y = np.ones(5)
        self.assertRaises(ValueError, Divide().transform, X, Y)
        # mismatched types
        X = np.array(['a','b', 'c'])
        Y = np.ones(3)
        self.assertRaises(TypeError, Divide().transform, X, Y)
        # proper case
        X = np.ones(10) * 4
        Y = np.ones(10) * 2
        assert np.array_equal(Divide().transform(X, Y), np.ones(10) * 2)
        # proper case >1D
        X = np.ones((5, 5)) * 4
        Y = np.ones((5, 5)) * 2
        assert np.array_equal(Divide().transform(X, Y), np.ones((5, 5)) * 2)


class test_StaticDot(unittest.TestCase):
    def test_transform(self):
        # mismatched shapes
        X = np.ones((5, 10))
        self.assertRaises(ValueError, StaticDot(np.ones((12, 10))).transform, X)
        # proper case
        X = np.ones(10)
        assert np.isclose(StaticDot(np.ones(10)).transform(X), 10.).all()
        # proper case >1D
        X = np.ones((2, 2))
        assert np.array_equal(StaticDot(np.ones((2, 3))).transform(X), (np.ones((2, 3)) * 2))


class test_Dot(unittest.TestCase):
    def test_transform(self):
        # mismatched_shapes
        X = np.ones((5, 10))
        Y = np.ones((12, 10))
        self.assertRaises(ValueError, Dot().transform, X, Y)
        # proper case
        X = np.ones(10)
        Y = np.ones(10)
        assert np.isclose(Dot().transform(X, Y), 10.).all()
        # proper case >1D
        X = np.ones((2, 2))
        Y = np.ones((2, 3))
        assert np.array_equal(Dot().transform(X, Y), np.ones((2, 3)) * 2)
        # multiple >1D
        Z = np.ones((3, 4))
        assert np.array_equal(Dot().transform(X, Y, Z), np.ones((2, 4)) * 6)


class test_Normalize(unittest.TestCase):
    def test_transform(self):
        # with zero
        X = np.zeros((100, 3)) + np.array([0, 1, 1])
        assert np.isclose(Normalize().transform(X), np.array([0., 0.5, 0.5])).all()
        # with inf
        X = np.zeros((100, 3)) + np.array([np.inf, 1, 1])
        self.assertRaises(ValueError, Normalize().transform, X)
        # all zero
        X = np.zeros((100, 5))
        assert np.isclose(Normalize().transform(X), np.ones(5) / 5).all()
        # proper case
        X = np.zeros((100, 4)) + np.arange(1, 5)
        assert np.isclose(Normalize().transform(X), np.array([0.1, 0.2, 0.3, 0.4])).all()
