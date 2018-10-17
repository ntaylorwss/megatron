import numpy as np
from .core import StatelessLayer, StatefulLayer


class Add(StatelessLayer):
    """Add up arrays element-wise."""
    def transform(self, *arrays):
        return np.sum(arrays, axis=0)


class Subtract(StatelessLayer):
    """Subtract one array from another."""
    def transform(self, X1, X2):
        return X1 - X2


class Multiply(StatelessLayer):
    """Multiply array by a given scalar.

    Parameters
    ----------
    factor : float
        multiplier.
    """
    def __init__(self, factor):
        super().__init__(factor=factor)

    def transform(self, X):
        return self.kwargs['factor'] * X


class Divide(StatelessLayer):
    """Divide given array by another given array element-wise.

    Parameters
    ----------
    impute : int/float or None
        the value to impute when encountering a divide by zero.
    """
    def __init__(self, impute=0):
        super().__init__(impute=impute)

    def transform(self, X1, X2):
        impute_array = np.ones_like(X1) * self.kwargs['impute']
        return np.divide(X1.astype(np.float16), X2, out=impute_array.astype(np.float16), where=X2!=0)


class Dot(StatelessLayer):
    """Multiply array by a given matrix, as matrix mulitplication.

    Parameters
    ----------
    W : np.array
        matrix by which to multiply.
    """
    def __init__(self, W):
        super().__init__(W=W)

    def transform(self, X):
        return np.dot(X, self.kwargs['W'])


class AddDim(StatelessLayer):
    """Add a dimension to an array.

    Parameters
    ----------
    axis : int
        the axis along which to place the new dimension.
    """
    def __init__(self, axis):
        super().__init__(axis=axis)

    def transform(self, X):
        return np.expand_dims(X, self.kwargs['axis'])


class OneHot(StatelessLayer):
    """One-hot encode an array for given range of values.

    Parameters
    ----------
    max_val : int
        maximum possible value.
    min_val : int (default: 0)
        minimum possible value.
    """
    def __init__(self, min_val=0, max_val=None):
        super().__init__(max_val=max_val, min_val=min_val)

    def transform(self, X):
        if not self.kwargs['max_val']:
            self.kwargs['max_val'] = X.max() + 1
        return (np.arange(self.kwargs['min_val'], self.kwargs['max_val']) == X[..., None]) * 1


class Reshape(StatelessLayer):
    """Reshape an array to a given new shape.

    Parameters
    ----------
    new_shape : tuple of int
        desired new shape for array.
    """
    def __init__(self, new_shape):
        super().__init__(new_shape=new_shape)

    def transform(self, X):
        return np.reshape(X, self.kwargs['new_shape'])
