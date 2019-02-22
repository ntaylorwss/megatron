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
    def __init__(self, dataframe, batch_size=32, exclude_cols=[]):
        self.dataframe = dataframe
        self.batch_size = batch_size
        if self.batch_size is None:
            self.n_batches = 1
        elif self.batch_size > 0:
            self.n_batches = math.ceil(self.dataframe.shape[0] / self.batch_size)
        else:
            raise ValueError("Batch size must be at least 1")
        self.n = 0
        self.exclude_cols = exclude_cols
        if len(set(self.exclude_cols) - set(self.dataframe.columns)) > 0:
            raise ValueError("Attempting to exclude a column not present in the data")

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
    def __init__(self, filepath, batch_size=32, exclude_cols=[]):
        self.filepath = filepath
        self.batch_size = batch_size
        self.cursor = self._make_generator()
        self.exclude_cols = exclude_cols
        with open(self.filepath, 'r') as f:
            data_cols = pd.read_csv(self.filepath, nrows=3).columns.values.tolist()
        if len(set(self.exclude_cols) - set(data_cols)) > 0:
            raise ValueError("Attempting to exclude a column not present in the data")

    def _make_generator(self):
        if self.batch_size is None:
            # must always be a generator, so for the full dataset, just yield it once
            def singular_generator():
                yield pd.read_csv(self.filepath)
            return singular_generator()
        elif self.batch_size > 0:
            return pd.read_csv(self.filepath, chunksize=self.batch_size)
        else:
            raise ValueError("Batch size must be at least 1")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            out = next(self.cursor).drop(self.exclude_cols, axis=1)
        except StopIteration:
            self.cursor = pd.read_csv(self.filepath, chunksize=self.batch_size)
            out = next(self.cursor).drop(self.exclude_cols, axis=1)
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
    def __init__(self, connection, query, batch_size=32, limit=None):
        self.connection = connection
        self.query = query
        self.batch_size = batch_size

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
