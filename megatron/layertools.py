import random
import numpy as np

# decorators

def vectorize(layer):
    class _vectorized_layer(layer):
        def __init__(self, *args, **kwargs):
            self.__class__.__name__ = layer.__name__
            super().__init__(*args, **kwargs)

        def transform(self, *inputs):
            return np.vectorize(super().transform)(*inputs)
    return _vectorized_layer


def map(layer):
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
    if isinstance(nodes, dict):
        return {key: layer(**layer_kwargs)(node) for key, node in nodes.items()}
    elif isinstance(nodes, list):
        return [layer(**layer_kwargs)(node) for node in nodes]
    else:
        raise ValueError("Can only apply a layer to multiple nodes")
