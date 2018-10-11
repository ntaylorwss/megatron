from .generator import *


def PandasData(dataframe, exclude_cols=[]):
    return next(PandasGenerator(dataframe, None, exclude_cols))


def CSVData(filepath, exclude_cols=[]):
    return next(CSVGenerator(filepath, None, exclude_cols))


def SQLData(connection, query):
    return next(SQLGenerator(connection, query, -1))
