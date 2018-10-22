import numpy as np
import pandas as pd
from .core import StatelessLayer, StatefulLayer


class AddDim(StatelessLayer):
    """Add a dimension to an array.

    Parameters
    ----------
    axis : int
        the axis along which to place the new dimension.
    """
    def __init__(self, axis):
        super().__init__(axis=axis)

    def transform(self, X):
        return np.expand_dims(X, self.kwargs['axis'])


class OneHotRange(StatefulLayer):
    """One-hot encode a numeric array where the values are a sequence."""
    def partial_fit(self, X):
        if self.metadata:
            self.metadata['min_val'] = min([self.metadata['min_val'], X.min()])
            self.metadata['max_val'] = max([self.metadata['max_val'], X.max()])
        else:
            self.metadata['min_val'] = X.min()
            self.metadata['max_val'] = X.max()

    def transform(self, X):
        return (np.arange(self.metadata['min_val'], self.metadata['max_val']) == X[..., None]) * 1


class OneHotLabels(StatefulLayer):
    """One-hot encode an array of categorical values, or non-consecutive numeric values."""
    def partial_fit(self, X):
        if self.metadata:
            self.metadata['categories'] = np.append(self.metadata['categories'], np.unique(X))
            self.metadata['categories'] = np.unique(self.metadata['categories'])
        else:
            self.metadata['categories'] = np.unique(X)

    def transform(self, X):
        return (self.metadata['categories'] == X[..., None]) * 1


class Reshape(StatelessLayer):
    """Reshape an array to a given new shape.

    Parameters
    ----------
    new_shape : tuple of int
        desired new shape for array.
    """
    def __init__(self, new_shape):
        super().__init__(new_shape=new_shape)

    def transform(self, X):
        return np.reshape(X, self.kwargs['new_shape'])


class SplitDict(StatelessLayer):
    def __init__(self, fields):
        super().__init__(n_outputs=len(fields), fields=fields)

    def transform(self, dicts):
        out = []
        as_df = pd.DataFrame(dicts.tolist())
        for key in self.kwargs['fields']:
            out.append(as_df[key].values)
        return out
