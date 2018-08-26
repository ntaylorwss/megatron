import numpy as np
from ..core import Transformation
from ..utils import initializer


class TimeSeries(Transformation):
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

    @initializer
    def __init__(self, window_size, time_axis=1, reverse=False):
        super().__init__()

    def transform(self, X):
        steps = [np.roll(X, i, axis=0) for i in range(self.window_size)]
        out = np.moveaxis(np.stack(steps), 0, self.time_axis)[self.window_size:]
        return np.flip(out, axis=-1) if self.reverse else out
