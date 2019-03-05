import numpy as np
import pandas as pd
from collections import Iterable
from .core import StatelessLayer


class Impute(StatelessLayer):
    """Replace instances of one data item with another, such as missing or NaN with zero.

    Parameters
    ----------
    imputation_dict : dict
        keys of the dictionary are targets to be replaced; values are corresponding replacements.
    """
    def __init__(self, imputation_dict):
        if not isinstance(imputation_dict, dict):
            raise TypeError("Impute layer takes dict as argument; keys are replaced by values")
        if len(imputation_dict) == 0:
            raise ValueError("Imputation dict must have at least one key-value pair")
        if any(isinstance(value, Iterable) and not isinstance(value, str)
               for value in imputation_dict.values()):
            raise TypeError("Values to be inserted must not be collections such as lists")
        super().__init__(imputation_dict=imputation_dict)

    def transform(self, X):
        for old, new in self.kwargs['imputation_dict'].items():
            if np.isnan(old):
                X[pd.isnull(X)] = new
            else:
                X[X==old] = new
        return X
