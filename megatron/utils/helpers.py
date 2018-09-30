import numpy as np


def column_stack(arrays):
    """Given a list of arrays, some of which are 2D, turn the 2D ones into 1D arrays."""
    out = []
    for array in arrays:
        if len(array.shape) == 1:
            out.append(array)
        elif len(array.shape) == 2:
            for column in list(array.T):
                out.append(column)
        else:
            raise ValueError("An array has more than 2 dimensions")
    return np.stack(out).T


def safe_divide(X1, X2, impute=0):
    impute_array = np.ones_like(X1) * impute
    return np.divide(X1.astype(np.float16), X2, out=impute_array.astype(np.float16), where=X2!=0)
