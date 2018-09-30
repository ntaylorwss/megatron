from ..nodes import InputNode
from ..nodes.wrappers import FeatureSet


def from_dataframe(df, eager=False, exclude_cols=[]):
    exclude_cols = set(exclude_cols)
    cols = [col for col in df.columns if not col in exclude_cols]
    nodes = [InputNode(name=col) for col in cols]
    if eager:
        nodes = [node(df[col].values) for col in cols]
    return FeatureSet(nodes)
