import numpy as np


def time_series(X, window_size=2, time_axis=1, reverse=False):
    steps = [np.roll(X, i, axis=0) for i in range(window_size)]
    out = np.moveaxis(np.stack(steps), 0, 1)[window_size:]
    return np.flip(out, axis=-1) if reverse else out
