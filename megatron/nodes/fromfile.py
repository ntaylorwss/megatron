import numpy as np
import pandas as pd
from .core import InputNode
from .wrappers import FeatureSet


def from_dataframe(df, exclude_cols=[], eager=False, nrows=None):
    """Load Input nodes from columns of a Pandas dataframe.

    Parameters
    ----------
    df : Pandas.DataFrame
        dataframe from which to load columns.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    eager : bool
        whether to load data as well, making for eager execution.
    nrows : int
        number of rows to load when eager is True. Default is for all rows to load.
    """
    exclude_cols = set(exclude_cols)
    cols = [col for col in df.columns if not col in exclude_cols]
    nodes = [InputNode(col) for col in cols]
    if eager:
        nodes = [node(df[col].values[:nrows]) for node, col in zip(nodes, cols)]
    return FeatureSet(nodes)


def from_csv(filepath, exclude_cols=[], eager=False, nrows=None):
    """Load Input nodes from columns of a CSV file.

    Parameters
    ----------
    filepath : str
        path of CSV file to be loaded.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    eager : bool
        whether to load data as well, making for eager execution.
    nrows : int
        number of rows to load when eager is True. Default is for all rows to load.
    """
    exclude_cols = set(exclude_cols)
    if eager:
        with open(filepath) as f:
            data = pd.read_csv(f, nrows=nrows)
            cols = data.columns
            nodes = [InputNode(col)(data[col].values) for col in cols if not col in exclude_cols]
    else:
        with open(filepath) as f:
            cols = pd.read_csv(f, nrows=2).columns
            nodes = [InputNode(col) for col in cols if not col in exclude_cols]
    return FeatureSet(nodes)


def from_sql(connection, query, eager=False, nrows=None):
    """Load Input nodes from columns of a Pandas dataframe.

    Parameters
    ----------
    connection : Connection
        database connection to load from.
    query : str
        query to execute in connection to load columns.
    eager : bool
        whether to load data as well, making for eager execution.
    nrows : int
        number of rows to load when eager is True. Default is for all rows to load.
    """
    col_query = query + ' LIMIT 2'
    cursor = connection.execute(col_query)
    cols = [col[0] for col in cursor.description]
    nodes = [InputNode(col) for col in cols]

    if eager:
        if nrows:
            query += ' LIMIT {}'.format(nrows)
        data = np.array(cursor.execute(query).fetchall()).T
        nodes = [node(datacol) for node, datacol in zip(nodes, data)]

    return FeatureSet(nodes)
