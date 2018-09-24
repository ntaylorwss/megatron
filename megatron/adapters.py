from .nodes import InputNode
from .nodes.wrappers import FeatureSet


def from_dataframe(df, pipeline, eager=False, exclude_cols=[]):
    exclude_cols = set(exclude_cols)
    cols = [col for col in df.columns if not col in exclude_cols]
    if eager:
        nodes = [Input(pipeline=pipeline, name=col)(df[col].values) for col in cols]
    else:
        nodes = [Input(pipeline=pipeline, name=col) for col in cols]
    return FeatureSet(nodes)
