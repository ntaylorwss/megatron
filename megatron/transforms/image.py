import numpy as np
from ..core import Transformation
from ..utils import initializer
try:
    import skimage
except ImportError:
    pass


class RGBtoGrey(Transformation):
    """Convert an RGB array representation of an image to greyscale.

    Parameters
    ----------
    method : {'luminosity', 'lightness', 'average'}
    """
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
    """Convert image to binary mask where a 1 indicates a non-black cell."""
    def transform(self, X):
        return (X.max(axis=2) > 0).astype(int)[:, :, np.newaxis]


class Downsample(Transformation):
    """Shrink an image to a given size proportionally.

    Parameters
    ----------
    new_shape : tuple of int
        the target shape for the new image.
    """
    @initializer
    def __init__(self, new_shape):
        super().__init__()

    def transform(self, X):
        if any(self.new_shape[i] > X.shape[i] for i in range(len(self.new_shape))):
            raise ValueError("New shape is larger than current in at least one dimension.")
        return skimage.transform.resize(X, self.new_shape)


class Upsample(Transformation):
    """Expand an image to a given size proportionally.

    Parameters
    ----------
    new_shape : tuple of int
        the target shape for the new image.
    """
    @initializer
    def __init__(self, new_shape):
        super().__init__()

    def transform(self, X):
        if any(self.new_shape[i] < X.shape[i] for i in range(len(self.new_shape))):
            raise ValueError("New shape is smaller than current in at least one dimension.")
        return skimage.transform.resize(X, self.new_shape)
