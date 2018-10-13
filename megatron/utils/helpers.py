import numpy as np


def safe_divide(X1, X2, impute=0):
    """Divide one array by another, while imputing the ones that are dividing by zero.

    Parameters
    ----------
    X1 : Numpy array
        numerator array.
    X2 : Numpy array
        denominator array.
    impute : int or float
        value to place in those elements which are dividing by zero.
    """
    impute_array = np.ones_like(X1) * impute
    return np.divide(X1.astype(np.float16), X2, out=impute_array.astype(np.float16), where=X2!=0)
