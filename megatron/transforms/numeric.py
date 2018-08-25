import numpy as np
from ..core import Transformation
from ..utils import initializer


class Whiten(Transformation):
    def fit(self, X):
        #self.metadata['mean'] = X.mean(axis=0)
        self.metadata['sd'] = X.std(axis=0)

    def transform(self, X):
        return (X - self.metadata['mean']) / self.metadata['sd']


class Add(Transformation):
    def transform(self, *arrays):
        return np.add(*arrays)


class Multiply(Transformation):
    @initializer
    def __init__(self, factor):
        super().__init__()

    def transform(self, X):
        return self.factor * X


class Dot(Transformation):
    @initializer
    def __init__(self, W):
        super().__init__()

    def transform(self, X):
        return np.dot(self.W, X)


class AddDim(Transformation):
    @initializer
    def __init__(self, axis):
        super().__init__()

    def transform(self, X):
        return np.expand_dims(X, self.axis)


class OneHot(Transformation):
    @initializer
    def __init__(self, max_val, min_val=0):
        super().__init__()

    def transform(self, X):
        if not self.max_val:
            self.max_val = X.max() + 1
        return (np.arange(self.min_val, self.max_val) == X[..., None]) * 1


class Reshape(Transformation):
    @initializer
    def __init__(self, new_shape):
        super().__init__()

    def transform(self, X):
        return np.reshape(X, self.new_shape)
