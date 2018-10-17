import numpy as np
from .core import StatelessLayer


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


class Concatenate(StatelessLayer):
    """Combine Nodes, creating n-length array for each observation."""
    def transform(self, *arrays):
        return np.stack(arrays).T
