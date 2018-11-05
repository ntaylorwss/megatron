import random
import numpy as np

# decorators

def vectorize(layer):
    """Cause a layer's transformation to be mapped to each row in the input data.

    Equivalent to numpy.vectorize.

    Parameters
    ----------
    layer : megatron.Layer
        Layer object to be wrapped.
    """
    class _vectorized_layer(layer):
        def __init__(self, *args, **kwargs):
            self.__class__.__name__ = layer.__name__
            super().__init__(*args, **kwargs)

        def transform(self, *inputs):
            return np.vectorize(super().transform)(*inputs)
    return _vectorized_layer


def map(layer):
    """Cause a layer's transformation to be mapped to each node called on.

    Parameters
    ----------
    layer : megatron.Layer
        Layer object to be wrapped.
    """
    class _mapped_layer(layer):
        def __init__(self, *args, **kwargs):
            self.__class__.__name__ = layer.__name__
            super().__init__(*args, **kwargs)

        def __call__(self, nodes):
            if isinstance(nodes, dict):
                return {key: super(layer, self).__call__(node) for key, node in nodes.items()}
            elif isinstance(nodes, list):
                return [super(layer, self).__call__(node) for node in nodes]
            else:
                raise ValueError("Layer can only be mapped when provided multiple nodes")
    return _mapped_layer


def maybe(p):
    """Cause a layer's transformation to operate probabilistically.

    Layers wrapped this way can operate only on one node.

    Parameters
    ----------
    p : float
        the probability with which the transformation should be applied.
        on negative trials, the data will simply pass through the layer.
    """
    def _maybe_layer(layer):
        # ensure layer only takes one argument
        n_args = len(inspect.signature(layer.transform).parameters)
        if n_args > 2:
            raise ValueError("Maybe decorator only works on layers who accept a single input node")

        class _maybe_layer_class(layer):
            def __init__(self, *args, **kwargs):
                self.__class__.__name__ = layer.__name__
                super().__init__(*args, **kwargs)

            def transform(self, X):
                if random.random() <= p:
                    return super().transform(X)
                else:
                    return X
        return _maybe_layer_class
    return _maybe_layer

# functions

def apply(layer, nodes, **layer_kwargs):
    """Create an instance of the Layer for each given Node with the same hyperparameters.

    The difference between map and apply is that map shares the Layer between Nodes,
    whereas apply creates separate Layers for each Node.

    Parameters
    ----------
    layer : megatron.Layer
        the Layer to apply to the Nodes.
    nodes : list of megatron.Node
        the Nodes to which the Layer should be applied.
    **layer_kwargs : kwargs
        the kwargs to be passed to the initialization of the Layer.
    """
    if isinstance(nodes, dict):
        return {key: layer(**layer_kwargs)(node) for key, node in nodes.items()}
    elif isinstance(nodes, list):
        return [layer(**layer_kwargs)(node) for node in nodes]
    else:
        raise ValueError("Can only apply a layer to multiple nodes")
