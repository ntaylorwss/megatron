import random
import numpy as np
import copy


class _mapped_call:
    def __init__(self, call):
        self.call = call

    def __call__(self, nodes):
        if isinstance(nodes, dict):
            return {key: self.call(node) for key, node in nodes.items()}
        elif isinstance(nodes, list):
            return [self.call(node) for node in nodes]
        else:
            raise ValueError("Layer can only be mapped when provided multiple nodes")


def map(layer):
    """Cause a layer's transformation to be mapped to each node called on.

    Parameters
    ----------
    layer : megatron.Layer
        Layer object to be wrapped.
    """
    layer._call = _mapped_call(layer._call)
    return layer


class _vectorized_func:
    def __init__(self, func):
        self.func = func

    def __call__(self, X):
        return np.array([self.func(row) for row in X])


def vectorize(layer):
    """Cause a layer's transformation to be mapped to each row in the input data.

    Parameters
    ----------
    layer : megatron.Layer
        Layer object to be wrapped.
    """
    layer.transform = _vectorized_func(layer.transform)
    return layer


class _probabilistic_func:
    def __init__(self, func, p):
        self.func = func
        self.p = p

    def __call__(self, X):
        if random.random() <= self.p:
            return self.func(X)
        else:
            return X


def maybe(layer, p):
    """Cause a layer's transformation to operate probabilistically.

    Layers wrapped this way can operate only on one node.

    Parameters
    ----------
    layer : megatron.Layer
        Layer object to be wrapped.
    p : float
        the probability with which the transformation should be applied.
        on negative trials, the data will simply pass through the layer.
    """
    layer.transform = _probabilistic_func(layer.transform)
    return layer
