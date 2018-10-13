from .generator import *


def PandasData(dataframe, exclude_cols=[]):
    """Load full Pandas dataframe in pipeline Input format.

    Parameters
    ----------
    dataframe : Pandas.DataFrame
        the dataframe to be used.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    """
    return next(PandasGenerator(dataframe, None, None, exclude_cols))


def CSVData(filepath, exclude_cols=[]):
    """Load full dataset from CSV file in pipeline Input format.

    Parameters
    ----------
    filepath : str
        the CSV filepath to be loaded from.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    """
    return next(CSVGenerator(filepath, None, None, exclude_cols))


def SQLData(connection, query):
    """Query full dataset from SQL database connection, in pipeline Input format.

    Parameters
    ----------
    connection : Connection
        a database connection to any valid SQL database engine.
    query : str
        a valid SQL query according to the engine being used, that extracts the data for Inputs.
    """
    return next(SQLGenerator(connection, query, -1))
