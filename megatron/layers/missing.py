from .core import StatelessLayer
from ..utils import initializer
import numpy as np
import pandas as pd


class Impute(StatelessLayer):
    def __init__(self, imputation_dict, name=None):
        super().__init__(name, imputation_dict=imputation_dict)

    def transform(self, X):
        for old, new in self.kwargs['imputation_dict'].items():
            if np.isnan(old):
                X[pd.isnull(X)] = new
            else:
                X[X==old] = new
        return X
