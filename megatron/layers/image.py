import numpy as np
import warnings
from .core import StatelessLayer
try:
    import skimage.transform
except ImportError:
    pass


class RGBtoGrey(StatelessLayer):
    """Convert an RGB array representation of an image to greyscale.

    Parameters
    ----------
    method : {'luminosity', 'lightness', 'average'}
    """
    def __init__(self, method='luminosity', keep_dim=False):
        super().__init__(method=method, keep_dim=keep_dim)

    def transform(self, X):
        if len(X.shape) != 3:
            raise ValueError("Input image must have 3 dimensions")

        if self.kwargs['method'] == 'lightness':
            out = (X.max(axis=2) + X.min(axis=2)) / 2.
        elif self.kwargs['method'] == 'average':
            out = X.mean(axis=2)
        elif self.kwargs['method'] == 'luminosity':
            out = np.matmul(X, np.array([0.21, 0.72, 0.07]))
        else:
            raise ValueError("Invalid averaging method for rgb_to_grey.")

        if self.kwargs['keep_dim']:
            out = np.expand_dims(out, -1)
        return out


class RGBtoBinary(StatelessLayer):
    """Convert image to binary mask where a 1 indicates a non-black cell.

    Parameters
    ----------
    keep_dim : bool
        if True, resulting image will stay 3D and will have 1 color channel. Otherwise 2D.
    """
    def __init__(self, keep_dim=True):
        super().__init__(keep_dim=keep_dim)

    def transform(self, X):
        out = (X.max(axis=2) > 0).astype(int)
        if self.kwargs['keep_dim']:
            out = out[:, :, np.newaxis]
        return out


class Downsample(StatelessLayer):
    """Shrink an image to a given size proportionally.

    Parameters
    ----------
    new_shape : tuple of int
        the target shape for the new image.
    """
    def __init__(self, new_shape):
        super().__init__(new_shape=new_shape)

    def transform(self, X):
        if len(self.kwargs['new_shape']) != len(X.shape):
            raise ValueError("New shape must be same number of dimensions as current shape.")
        elif any(kw_dim > data_dim for kw_dim, data_dim in zip(self.kwargs['new_shape'], X.shape)):
            raise ValueError("New shape is larger than current in at least one dimension.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = skimage.transform.resize(X, self.kwargs['new_shape'])
        return out


class Upsample(StatelessLayer):
    """Expand an image to a given size proportionally.

    Parameters
    ----------
    new_shape : tuple of int
        the target shape for the new image.
    """
    def __init__(self, new_shape):
        super().__init__(new_shape=new_shape)

    def transform(self, X):
        if len(self.kwargs['new_shape']) != len(X.shape):
            raise ValueError("New shape must be same number of dimensions as current shape.")
        elif any(kw_dim < data_dim for kw_dim, data_dim in zip(self.kwargs['new_shape'], X.shape)):
            raise ValueError("New shape is smaller than current in at least one dimension.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = skimage.transform.resize(X, self.kwargs['new_shape'])
        return out
