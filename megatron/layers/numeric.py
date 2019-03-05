import numpy as np
from .core import StatelessLayer, StatefulLayer


class Add(StatelessLayer):
    """Add up arrays element-wise."""
    def __init__(self):
        super().__init__()

    def transform(self, *arrays):
        if len(set([a.shape for a in arrays])) > 1:
            raise ValueError("Arrays must all be same shape to be added")
        return np.sum(arrays, axis=0)


class Subtract(StatelessLayer):
    """Subtract one array from another."""
    def transform(self, X1, X2):
        return X1 - X2


class ScalarMultiply(StatelessLayer):
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


class ElementWiseMultiply(StatelessLayer):
    """Multiply two same-sized arrays element-by-element."""
    def transform(self, X, Y):
        return X * Y


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


class StaticDot(StatelessLayer):
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


class Dot(StatelessLayer):
    """Multiply multiple arrays together as matrix multiplication."""
    def transform(self, *arrays):
        return np.linalg.multi_dot(arrays)


class Normalize(StatelessLayer):
    """Divide array by total to cause it to sum to one. If zero array, make uniform."""
    def transform(self, X):
        if len(X.shape) != 2:
            raise ValueError("Data must be 2-dimensional")
        out = X.copy()
        S = out.sum(axis=1)
        if np.isinf(S).any():
            raise ValueError("Sum of at least one observation is infinity, cannot normalize")
        out[S<0.0001, :] = np.ones(out.shape[1])
        return out / out.sum(axis=1, keepdims=True)
