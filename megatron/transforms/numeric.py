import numpy as np
from ..core import Transformation
from ..utils import initializer


class Whiten(Transformation):
    """Bring mean to 0 and standard deviation to 1."""
    def fit(self, X):
        self.metadata['mean'] = X.mean(axis=0)
        self.metadata['sd'] = X.std(axis=0)

    def transform(self, X):
        return (X - self.metadata['mean']) / self.metadata['sd']


class Add(Transformation):
    """Add up arrays element-wise."""
    def transform(self, *arrays):
        return np.add(*arrays)


class Multiply(Transformation):
    """Multiply array by a given scalar.

    Parameters
    ----------
    factor : float
        multiplier.
    """
    @initializer
    def __init__(self, factor):
        super().__init__()

    def transform(self, X):
        return self.factor * X


class Dot(Transformation):
    """Multiply array by a given matrix, as matrix mulitplication.

    Parameters
    ----------
    W : np.array
        matrix by which to multiply.
    """
    @initializer
    def __init__(self, W):
        super().__init__()

    def transform(self, X):
        return np.dot(self.W, X)


class AddDim(Transformation):
    """Add a dimension to an array.

    Parameters
    ----------
    axis : int
        the axis along which to place the new dimension.
    """
    @initializer
    def __init__(self, axis):
        super().__init__()

    def transform(self, X):
        return np.expand_dims(X, self.axis)


class OneHot(Transformation):
    """One-hot encode an array for given range of values.

    Parameters
    ----------
    max_val : int
        maximum possible value.
    min_val : int (default: 0)
        minimum possible value.
    """
    @initializer
    def __init__(self, max_val, min_val=0):
        super().__init__()

    def transform(self, X):
        if not self.max_val:
            self.max_val = X.max() + 1
        return (np.arange(self.min_val, self.max_val) == X[..., None]) * 1


class Reshape(Transformation):
    """Reshape an array to a given new shape.

    Parameters
    ----------
    new_shape : tuple of int
        desired new shape for array.
    """
    @initializer
    def __init__(self, new_shape):
        super().__init__()

    def transform(self, X):
        return np.reshape(X, self.new_shape)
