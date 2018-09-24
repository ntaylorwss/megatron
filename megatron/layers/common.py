import numpy as np
from .core import StatelessLayer
from ..utils import initializer


class TimeSeries(StatelessLayer):
    """Adds a time dimension to a dataset by rolling a window over the data.

    Parameters
    ----------
    window_size : int
        length of the window; number of timesteps in the time series.
    time_axis : int
        on which axis in the array to place the time dimension.
    reverse : bool
        if True, oldest data is first; if False, newest data is first.
    """
    def __init__(self, window_size, time_axis=1, reverse=False, name=None):
        if name is None:
            name = 'window({})'.format(window_size)
        super().__init__(name, window_size=window_size, time_axis=time_axis, reverse=reverse)

    def transform(self, X):
        steps = [np.roll(X, i, axis=0) for i in range(self.kwargs['window_size'])]
        out = np.moveaxis(np.stack(steps), 0, self.kwargs['time_axis'])[self.kwargs['window_size']:]
        return np.flip(out, axis=-1) if self.kwargs['reverse'] else out


class Retype(StatelessLayer):
    """Re-defines the data type for a Numpy array's contents.

    Parameters
    ----------
    new_type : type
        the new type for the array to be cast to.
    """
    def __init__(self, new_type, name=None):
        if name is None:
            name = new_type.__name__
        super().__init__(name, new_type=new_type)

    def transform(self, X):
        return X.astype(self.kwargs['new_type'])
