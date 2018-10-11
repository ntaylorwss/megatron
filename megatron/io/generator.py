import math
import numpy as np
import pandas as pd


class DataGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class PandasGenerator(DataGenerator):
    def __init__(self, dataframe, batch_size, exclude_cols=[]):
        super().__init__(batch_size)
        self.dataframe = dataframe
        self.n = 0
        if self.batch_size:
            self.n_batches = math.ceil(self.dataframe.shape[0] / self.batch_size)
        else:
            self.n_batches = 1
        self.exclude_cols = exclude_cols

    def __next__(self):
        if self.n == self.n_batches:
            self.n = 0
            raise StopIteration()

        if self.batch_size:
            start = self.n * self.batch_size
            end = min([self.dataframe.shape[0], start + self.batch_size])
            out = self.dataframe.iloc[start:end].drop(self.exclude_cols, axis=1)
        else:
            out = self.dataframe.drop(self.exclude_cols, axis=1)
        self.n += 1

        return dict(zip(out.columns, out.T.values))


class CSVGenerator(DataGenerator):
    def __init__(self, filepath, batch_size, exclude_cols=[]):
        super().__init__(batch_size)
        self.filepath = filepath
        # take advantage of Pandas read_csv function to make it simpler and more robust
        self.cursor = pd.read_csv(self.filepath, chunksize=self.batch_size)
        self.exclude_cols = exclude_cols

    def __next__(self):
        try:
            new_df = next(self.cursor).drop(self.exclude_cols, axis=1)
        except StopIteration:
            self.cursor = pd.read_csv(self.filepath, chunksize=self.batch_size)
            raise
        return dict(zip(new_df.columns, new_df.T.values))


class SQLGenerator(DataGenerator):
    def __init__(self, connection, query, batch_size, limit=None):
        super().__init__(batch_size)
        self.connection = connection
        self.query = query
        if limit:
            self.query += ' LIMIT {}'.format(nrows)
        self.cursor = self.connection.execute(self.query)
        self.names = [col[0] for col in self.cursor.description]

    def __next__(self):
        out = self.cursor.fetchmany(self.batch_size)
        coldata = np.array(out).T
        return dict(zip(self.names, coldata))
