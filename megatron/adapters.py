from .core import Input, FeatureSet


def from_dataframe(df, graph, eager=False, exclude_cols=[]):
    exclude_cols = set(exclude_cols)
    cols = [col for col in df.columns if not col in exclude_cols]
    if eager:
        nodes = [Input(graph=graph, name=col)(df[col].values) for col in cols]
    else:
        nodes = [Input(graph=graph, name=col) for col in cols]
    return FeatureSet(nodes, cols)
