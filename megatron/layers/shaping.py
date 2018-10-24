import numpy as np
import pandas as pd
from .core import StatelessLayer, StatefulLayer


class Cast(StatelessLayer):
    """Re-defines the data type for a Numpy array's contents.

    Parameters
    ----------
    new_type : type
        the new type for the array to be cast to.
    name : str (default: None)
        name for the layer. If None, defaults to name of new type.
    """
    def __init__(self, new_type, name=None):
        super().__init__(new_type=new_type)
        self.name = name if name else new_type.__name__

    def transform(self, X):
        return X.astype(self.kwargs['new_type'])


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
        return (np.arange(self.metadata['min_val'], self.metadata['max_val']+1) == X[..., None]) * 1


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


class TimeSeries(StatelessLayer):
    """Adds a time dimension to a dataset by rolling a window over the data.

    Parameters
    ----------
    window_size : int
        length of the window; number of timesteps in the time series.
    time_axis : int
        on which axis in the array to place the time dimension.
    reverse : bool (default: False)
        if True, oldest data is first; if False, newest data is first.
    name : str (default: None)
        name for the layer. If None, defaults to name of class.
    """
    def __init__(self, window_size, time_axis=1, reverse=False, name=None):
        super().__init__(window_size=window_size, time_axis=time_axis, reverse=reverse)
        self.name = 'window({})'.format(window_size)

    def transform(self, X):
        steps = [np.roll(X, i, axis=0) for i in range(self.kwargs['window_size'])]
        out = np.moveaxis(np.stack(steps), 0, self.kwargs['time_axis'])[self.kwargs['window_size']:]
        return np.flip(out, axis=-1) if self.kwargs['reverse'] else out


class Concatenate(StatelessLayer):
    """Combine Nodes, creating n-length array for each observation."""
    def transform(self, *arrays):
        arrays = list(arrays)
        for i, a in enumerate(arrays):
            if len(a.shape) == 1:
                arrays[i] = np.expand_dims(a, -1)
        return np.hstack(arrays)
