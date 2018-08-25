import numpy as np
from ..core import Transformation
from ..utils import initializer
try:
    import skimage
except ImportError:
    pass


class RGBtoGrey(Transformation):
    @initializer
    def __init__(self, method='luminosity'):
        super().__init__()

    def transform(self, X):
        if self.method == 'lightness':
            return (X.max(axis=2) + X.min(axis=2)) / 2.
        elif self.method == 'average':
            return X.mean(axis=2)
        elif self.method == 'luminosity':
            return np.matmul(X, np.array([0.21, 0.72, 0.07]))
        else:
            raise ValueError("Invalid averaging method for rgb_to_grey.")


class RGBtoBinary(Transformation):
    def transform(self, X):
        return (X.max(axis=2) > 0).astype(int)[:, :, np.newaxis]


class Downsample(Transformation):
    @initializer
    def __init__(self, new_shape):
        super().__init__()

    def transform(self, X):
        if any(self.new_shape[i] > X.shape[i] for i in range(len(self.new_shape))):
            raise ValueError("New shape is larger than current in at least one dimension.")
        return skimage.transform.resize(X, self.new_shape)


class Upsample(Transformation):
    @initializer
    def __init__(self, new_shape):
        super().__init__()

    def transform(self, X):
        if any(self.new_shape[i] < X.shape[i] for i in range(len(self.new_shape))):
            raise ValueError("New shape is smaller than current in at least one dimension.")
        return resize(X, self.new_shape)
