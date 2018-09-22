from ..core import Transformation
from ..utils import initializer
import numpy as np
import pandas as pd


class Impute(Transformation):
    @initializer
    def __init__(self, imputation_dict, name=None):
        super().__init__(name=name)

    def transform(self, X):
        for old, new in self.imputation_dict.items():
            if np.isnan(old):
                X[pd.isnull(X)] = new
            else:
                X[X==old] = new
        return X
