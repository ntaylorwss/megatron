import numpy as np
from ..core import Transformer
try:
    import skimage
except ImportError:
    pass


class RGBtoGrey(Transformer):
    def transform(self, X):
        if self.kwargs['method'] == 'lightness':
            return (X.max(axis=2) + X.min(axis=2)) / 2.
        elif self.kwargs['method'] == 'average':
            return X.mean(axis=2)
        elif self.kwargs['method'] == 'luminosity':
            return np.matmul(X, np.array([0.21, 0.72, 0.07]))
        else:
            raise ValueError("Invalid averaging method for rgb_to_grey.")


class RGBtoBinary(Transformer):
    def transform(self, X):
        return (X.max(axis=2) > 0).astype(int)[:, :, np.newaxis]


class Downsample(Transformer):
    def transform(self, X):
        if any(new_shape[i] > X.shape[i] for i in range(len(new_shape))):
            raise ValueError("New shape is larger than current in at least one dimension.")
        return skimage.transform.resize(X, new_shape)


class Upsample(Transformer):
    def transform(self, X):
        if any(new_shape[i] < X.shape[i] for i in range(len(new_shape))):
            raise ValueError("New shape is smaller than current in at least one dimension.")
        return resize(X, new_shape)
