import numpy as np
from ..core import Transformer


class TimeSeries(Transformer):
    def transform(self, X):
        steps = [np.roll(X, i, axis=0) for i in range(self.kwargs['window_size'])]
        out = np.moveaxis(np.stack(steps), 0, self.kwargs['time_axis'])[self.kwargs['window_size']]
        return np.flip(out, axis=-1) if reverse else out
