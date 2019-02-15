import math
import numpy as np
import pandas as pd
from ..utils.generic import listify


class PandasGenerator:
    """A generator of data batches from a Pandas Dataframe into Megatron Input nodes.

    Parameters
    ----------
    dataframe : Pandas.DataFrame
        dataframe to load data from.
    batch_size : int
        number of observations to yield in each iteration.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    """
    def __init__(self, dataframe, batch_size, exclude_cols=[]):
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.n = 0
        if self.batch_size:
            self.n_batches = math.ceil(self.dataframe.shape[0] / self.batch_size)
        else:
            self.n_batches = 1
        self.exclude_cols = exclude_cols

    def __iter__(self):
        return self

    def __next__(self):
        if self.n == self.n_batches:
            self.n = 0

        if self.batch_size:
            start = self.n * self.batch_size
            end = min([self.dataframe.shape[0], start + self.batch_size])
            out = self.dataframe.iloc[start:end].drop(self.exclude_cols, axis=1)
        else:
            out = self.dataframe.drop(self.exclude_cols, axis=1)
        self.n += 1

        return dict(zip(out.columns, out.values.T))


class CSVGenerator:
    """A generator of data batches from a CSV file in pipeline Input format.

    Parameters
    ----------
    filepath : str
        the CSV filepath to be loaded from.
    batch_size : int
        number of observations to yield in each iteration.
    exclude_cols : list of str (default: [])
        any columns that should not be loaded as Input.
    """
    def __init__(self, filepath, batch_size, exclude_cols=[]):
        self.filepath = filepath
        self.batch_size = batch_size
        # take advantage of Pandas read_csv function to make it simpler and more robust
        if self.batch_size:
            self.cursor = pd.read_csv(self.filepath, chunksize=self.batch_size)
        else:
            self.cursor = self._make_generator()
        self.exclude_cols = exclude_cols

    def _make_generator(self):
        yield pd.read_csv(self.filepath)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out = next(self.cursor).drop(self.exclude_cols, axis=1)
        except StopIteration:
            self.cursor = pd.read_csv(self.filepath, chunksize=self.batch_size)
        return dict(zip(out.columns, out.values.T))


class SQLGenerator:
    """A generator of data batches from a SQL query in pipeline Input format.

    Parameters
    ----------
    connection : Connection
        a database connection to any valid SQL database engine.
    query : str
        a valid SQL query according to the engine being used, that extracts the data for Inputs.
    batch_size : int
        number of observations to yield in each iteration.
    limit : int
        number of observations to use from the query in total.
    """
    def __init__(self, connection, query, batch_size, limit=None):
        self.batch_size = batch_size
        self.connection = connection
        self.query = query
        if limit:
            self.query += ' LIMIT {}'.format(nrows)
        self.cursor = self.connection.execute(self.query)
        self.names = [col[0] for col in self.cursor.description]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out = self.cursor.fetchmany(self.batch_size)
        except StopIteration:
            self.cursor = self.connection.execute(self.query)
            out = self.cursor.fetchmany(self.batch_size)
        coldata = np.array(out).T

        return dict(zip(self.names, coldata))
