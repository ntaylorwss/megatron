from .generator import *


def PandasData(dataframe, exclude_cols=[], nrows=None):
    """Load fixed data from Pandas Dataframe into Megatron Input nodes, one for each column.

    Parameters
    ----------
    dataframe : Pandas.DataFrame
        the dataframe to be used.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    nrows : int (default: None)
        number of rows to load. If None, loads all rows.
    """
    return next(PandasGenerator(dataframe, nrows, exclude_cols))


def CSVData(filepath, exclude_cols=[], nrows=None):
    """Load fixed data from CSV filepath into Megatron Input nodes, one for each column.

    Parameters
    ----------
    filepath : str
        the CSV filepath to be loaded from.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    nrows : int (default: None)
        number of rows to load. If None, load all rows.
    """
    return next(CSVGenerator(filepath, nrows, exclude_cols))


def SQLData(connection, query):
    """Load fixed data from SQL query into Megatron Input nodes, one for each column.

    Parameters
    ----------
    connection : Connection
        a database connection to any valid SQL database engine.
    query : str
        a valid SQL query according to the engine being used, that extracts the data for Inputs.
    """
    return next(SQLGenerator(connection, query, -1))
