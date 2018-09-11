from ..core import Input


def from_dataframe(df, graph, eager=False):
    if eager:
        out = {col: Input(graph=graph, name=col)(df[col].values) for col in df.columns}
    else:
        out = {col: Input(graph=graph, name=col) for col in df.columns}
    return out
