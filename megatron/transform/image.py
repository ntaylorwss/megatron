import skimage
import numpy as np


def rgb_to_grey(X, method='luminosity'):
    if method == 'lightness':
        return (X.max(axis=2) + X.min(axis=2)) / 2.
    elif method == 'average':
        return X.mean(axis=2)
    elif method == 'luminosity':
        return np.matmul(X, np.array([0.21, 0.72, 0.07]))
    else:
        raise ValueError("Invalid averaging method for rgb_to_grey.")


def rgb_to_binary(X):
    return (X.max(axis=2) > 0).astype(int)[:, :, np.newaxis]


def downsample(X, new_shape):
    if any(new_shape[i] > X.shape[i] for i in range(len(new_shape))):
        raise ValueError("New shape is larger than current in at least one dimension.")
    return skimage.transform.resize(X, new_shape)


def upsample(X, new_shape):
    if any(new_shape[i] < X.shape[i] for i in range(len(new_shape))):
        raise ValueError("New shape is smaller than current in at least one dimension.")
    return resize(X, new_shape)
