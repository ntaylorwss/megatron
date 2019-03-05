import numpy as np
import pandas as pd
from .core import StatelessLayer, StatefulLayer


class Cast(StatelessLayer):
    """Re-defines the data type for a Numpy array's contents.

    Parameters
    ----------
    new_type : type
        the new type for the array to be cast to.
    """
    def __init__(self, new_type):
        if not isinstance(new_type, type):
            raise TypeError("new_type must be a valid datatype")
        super().__init__(new_type=new_type)

    def transform(self, X):
        return X.astype(self.kwargs['new_type'])


class AddDim(StatelessLayer):
    """Add a dimension to an array.

    Parameters
    ----------
    axis : int
        the axis along which to place the new dimension.
    """
    def __init__(self, axis=-1):
        super().__init__(axis=axis)

    def transform(self, X):
        if self.kwargs['axis'] > len(X.shape):
            raise ValueError("Axis out of range for new dimension")
        return np.expand_dims(X, axis=self.kwargs['axis'])


class OneHotRange(StatefulLayer):
    """One-hot encode a numeric array where the values are a sequence."""
    def __init__(self, strict=False):
        super().__init__(strict=strict)

    def partial_fit(self, X):
        try:
            X.astype(int)
        except ValueError:
            raise TypeError("Array must be ints only")
        if not np.array_equal(X, X.astype(int)):
            raise TypeError("Array must be ints only")

        if self.metadata:
            self.metadata['min_val'] = min([self.metadata['min_val'], X.min()])
            self.metadata['max_val'] = max([self.metadata['max_val'], X.max()])
        else:
            self.metadata['min_val'] = X.min()
            self.metadata['max_val'] = X.max()

    def transform(self, X):
        out = (np.arange(self.metadata['min_val'], self.metadata['max_val']+1) == X[..., None]) * 1
        if self.kwargs['strict'] and (out.sum(axis=-1) == 0).any():
            raise ValueError("New value encountered in transform that was not present in fit")
        return out


class OneHotLabels(StatefulLayer):
    """One-hot encode an array of categorical values, or non-consecutive numeric values."""
    def __init__(self, strict=False):
        super().__init__(strict=strict)

    def partial_fit(self, X):
        if self.metadata:
            self.metadata['categories'] = np.append(self.metadata['categories'], np.unique(X))
            self.metadata['categories'] = np.unique(self.metadata['categories'])
        else:
            self.metadata['categories'] = np.unique(X)

    def transform(self, X):
        out = (self.metadata['categories'] == X[..., None]) * 1
        if self.kwargs['strict'] and (out.sum(axis=-1) == 0).any():
            raise ValueError("New value encountered in transform that was not present in fit")
        return out


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


class Flatten(StatelessLayer):
    """Reshape an array to be 1D."""
    def transform(self, X):
        return X.flatten()


class SplitDict(StatelessLayer):
    """Split dictionary data into separate nodes, with one node per key in the dictionary.

    Parameters
    ----------
    fields : list of str
        list of fields, dictionary keys, to be pulled out into their own nodes.
    """
    def __init__(self, fields):
        super().__init__(n_outputs=len(fields), fields=fields)

    def transform(self, dicts):
        out = []
        as_df = pd.DataFrame(dicts.tolist())
        for key in self.kwargs['fields']:
            out.append(as_df[key].values)
        return out


class TimeSeries(StatefulLayer):
    """Adds a time dimension to a dataset by rolling a window over the data.

    Parameters
    ----------
    window_size : int
        length of the window; number of timesteps in the time series.
    time_axis : int
        on which axis in the array to place the time dimension.
    reverse : bool (default: False)
        if True, oldest data is first; if False, newest data is first.
    """
    def __init__(self, window_size, time_axis=1, reverse=False):
        if window_size <= 1:
            raise ValueError("Window size must be greater than 1")
        super().__init__(window_size=window_size, time_axis=time_axis, reverse=reverse)

    def partial_fit(self, X):
        self.metadata['shape'] = list(X.shape[1:])
        self.metadata['previous'] = np.zeros([self.kwargs['window_size']-1] + self.metadata['shape'])

    def transform(self, X):
        internal = [np.roll(X, i, axis=0) for i in range(self.kwargs['window_size'])]
        internal = np.moveaxis(np.stack(internal), 0, self.kwargs['time_axis'])
        internal = internal[self.kwargs['window_size']:]
        begin = np.concatenate([X[:self.kwargs['window_size']], self.metadata['previous']])
        begin = [np.roll(begin, i, axis=0) for i in range(self.kwargs['window_size'])]
        begin = np.moveaxis(np.stack(begin), 0, self.kwargs['time_axis'])
        begin = begin[:self.kwargs['window_size']]

        self.metadata['previous'] = X[-self.kwargs['window_size']:]
        out = np.concatenate([begin, internal])
        if self.kwargs['reverse']:
            slices = [slice(None) for i in range(len(X.shape))]
            slices[self.kwargs['time_axis']] = slice(None, None, -1)
            out = out[tuple(slices)]
        return out


class Concatenate(StatelessLayer):
    """Combine arrays along a given axis. Does not create a new axis, unless all 1D inputs.

    Parameters
    ----------
    axis : int (default: -1)
        axis along which to concatenate arrays. -1 means the last axis.
    """
    def __init__(self, axis=-1):
        super().__init__(axis=axis)

    def transform(self, *arrays):
        arrays = list(arrays)
        for i, a in enumerate(arrays):
            if len(a.shape) == 1:
                arrays[i] = np.expand_dims(a, -1)
        return np.concatenate(arrays, axis=self.kwargs['axis'])


class Slice(StatelessLayer):
    """Apply Numpy array slicing. Each slice corresponds to a dimension.

    Slices (passed as hyperparameters) are constructed by the following procedure:
    - To get just N: provide the integer N as the slice
    - To slice from N to the end: provide a 1-tuple of the integer N, e.g. (5,).
    - To slice from M to N exclusive: provide a 2-tuple of the integers M and N, e.g. (3, 6).
    - To slice from M to N with skip P: provide a 3-tuple of the integers M, N, and P.

    Parameters
    ----------
    *slices : int(s) or tuple(s)
        the slices to be applied. Must not overlap. Formatting discussed above.
    """
    def __init__(self, *slices):
        if isinstance(slices, list):
            raise ValueError("Slices for each dimension are passed as separate arguments, not in a list")
        super().__init__(slices=slices)

    def transform(self, X):
        if len(X.shape) != len(self.kwargs['slices']):
            raise ValueError("Number of dimensions in X must match number of slices")
        new_slices = []
        for i, s in enumerate(self.kwargs['slices']):
            if s is None:
                new_slices.append(slice(None))
            elif isinstance(s, int):
                new_slices.append(s)
            elif len(s) == 1:
                new_slices.append(slice(s[0], None))
            else:
                new_slices.append(slice(*s))
        return X[tuple(new_slices)]


class Filter(StatelessLayer):
    """Apply given mask to given array along the first axis to filter out observations."""
    def transform(self, X, mask):
        if mask.sum() == 0:
            raise ValueError("Mask must retain at least one observation")
        return X[mask.astype(bool)]
