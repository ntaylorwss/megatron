import numpy as np
import pandas as pd
from .core import StatelessLayer


class Impute(StatelessLayer):
    """Replace instances of one data item with another, such as missing or NaN with zero.

    Parameters
    ----------
    imputation_dict : dict
        keys of the dictionary are targets to be replaced; values are corresponding replacements.
    """
    def __init__(self, imputation_dict):
        super().__init__(imputation_dict=imputation_dict)

    def transform(self, X):
        for old, new in self.kwargs['imputation_dict'].items():
            if np.isnan(old):
                X[pd.isnull(X)] = new
            else:
                X[X==old] = new
        return X
