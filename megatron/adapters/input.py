import pandas as pd
from ..nodes import InputNode
from ..nodes.wrappers import FeatureSet


def df_to_featureset(df, eager=False, exclude_cols=[]):
    exclude_cols = set(exclude_cols)
    cols = [col for col in df.columns if not col in exclude_cols]
    nodes = [InputNode(name=col) for col in cols]
    if eager:
        nodes = [node(df[col].values) for col in cols]
    return FeatureSet(nodes)


def csv_to_featureset(filename, eager=False, exclude_cols=[]):
    if eager:
        return df_to_featureset(pd.read_csv(filename), eager=True, exclude_cols=exclude_cols)
    else:
        with open(filename) as f:
            cols = pd.read_csv(filename, nrows=2).columns
        nodes = [InputNode(name=col) for col in cols]
        return FeatureSet(nodes)
