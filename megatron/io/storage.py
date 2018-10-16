import re
import sqlite3
import pickle
import numpy as np
import pandas as pd
from .. import utils


class DataStore:
    """SQL table of input data and output features, associated with a single pipeline.

    Parameters
    ----------
    table_name : str
        name of pipeline's cache table in the database.
    db_conn : Connection
        database connection to query.
    """
    def __init__(self, table_name, version, db_conn):
        self.db = db_conn
        if version:
            self.table_name = '{}_{}'.format(table_name, version)
        else:
            self.table_name = table_name
        self.output_names = []
        self.dtypes = {}
        self.original_shapes = {}

    def _check_schema(self, output_data):
        """If existing SQL colnames and data colnames are off, throw error that table is in use.

        Parameters
        ----------
        output_data : dict of ndarray
            resulting features from applying pipeline to input_data.
        """
        sql_cols = "SELECT sql FROM sqlite_master WHERE name='{}';"
        sql_cols = sql_cols.format(self.table_name)
        sql_cols = self.db.execute(sql_cols).fetchone()[0]
        sql_cols = np.array([c[1:-1] for c in re.findall('"[^\s]+"', sql_cols)])
        out_cols = np.insert(output_data.columns.values, 0, 'ind')
        if not np.array_equal(sql_cols, out_cols):
            msg = "Pipeline name already in use with different outputs: currently {}, not {}."
            msg = msg.format(sql_cols.tolist(), output_data.columns.values.tolist())
            raise ValueError(msg)

    def write(self, output_data, data_index):
        """Write set of observations to database.

        For features that are multi-dimensional, use pickle to compress to string.

        Parameters
        ----------
        output_data : dict of ndarray
            resulting features from applying pipeline to input_data.
        data_index : np.array
            index of observations.
        """
        # identify and pickle any multi-dimensional features
        output_df = pd.DataFrame()
        for k in output_data:
            if len(output_data[k].shape) > 1:
                self.dtypes[k] = 'TEXT'
                self.original_shapes[k] = output_data[k].shape[1:]
                flat_data = output_data[k].reshape((output_data[k].shape[0], -1))
                flat_data = np.apply_along_axis(lambda x: pickle.dumps(x), axis=1, arr=flat_data)
                output_df[k] = flat_data
            else:
                self.dtypes[k] = 'REAL'
                output_df[k] = output_data[k]

        # create table if it doesn't yet exist; if it does, check the schema matches the new data
        exists_query = 'SELECT name FROM sqlite_master WHERE type="table" AND name="{}";'
        exists_query = exists_query.format(self.table_name)
        if self.db.execute(exists_query).fetchone():
            self._check_schema(output_df)
        else:
            cols = ', '.join(['"{}" {}'.format(name, self.dtypes[name])
                               for name in output_df.columns])
            make_query = 'CREATE TABLE {} ("ind" VARCHAR, {});'
            self.db.execute(make_query.format(self.table_name, cols))
            self.db.commit()

        # delete data for indices that are being written
        inds = ','.join([str(i) for i in data_index.tolist()])
        self.db.execute("DELETE FROM {} WHERE ind IN ({})".format(self.table_name, inds))

        # write new data to db
        output_df['ind'] = data_index
        output_df.to_sql(self.table_name, self.db, if_exists='append', index=False)
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS ind ON {} (ind)".format(self.table_name))
        self.db.commit()

        # save output field names for future reference
        self.output_names = list(output_df.keys())

    def read(self, output_names=None, lookup=None):
        """Retrieve all processed features from cache, or lookup a single observation.

        For features that are multi-dimensional, use pickle to read string.

        Parameters
        ----------
        output_names : list of str
            names of output columns to retrieve. If none, get all outputs.
        lookup: dict of ndarray
            input data to lookup output for, in dictionary form.
        """
        if output_names:
            output_names = ', '.join(utils.generic.listify(output_names))
        else:
            output_names = ', '.join(self.output_names)
        query = "SELECT {}, ind FROM {} ".format(output_names, self.table_name)

        if lookup:
            lookup = utils.generic.listify(lookup)
            query += "WHERE ind IN ({})".format(','.join(lookup))

        out = pd.read_sql_query(query, self.db, index_col='ind')
        out = dict(zip(out.columns, out.values.T))
        for k in out:
            if self.dtypes[k] == 'TEXT':
                out[k] = np.array([np.loads(v) for v in out[k]])
                out[k] = out[k].reshape([-1] + list(self.original_shapes[k]))
        return out
