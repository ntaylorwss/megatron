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
    version : str
        version tag for pipeline's cache table in the database.
    db_conn : Connection
        database connection to query.
    """
    def __init__(self, table_name, version, db_conn, overwrite):
        self.db = db_conn
        if version:
            self.table_name = '{}_{}'.format(table_name, version)
        else:
            self.table_name = table_name
        if overwite:
            self.db.execute("DROP TABLE IF EXISTS {}".format(self.table_name))

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
        self.dtypes = []
        self.original_shapes = []
        output_df = pd.DataFrame()

        # identify and pickle any multi-dimensional features
        for i, out_data in enumerate(output_data):
            self.original_shapes.append(out_data.shape[1:])
            if len(out_data.shape) > 1:
                self.dtypes.append('TEXT')
                flat_data = out_data.reshape((out_data.shape[0], -1))
                flat_data = np.apply_along_axis(lambda x: pickle.dumps(x), axis=1, arr=flat_data)
                output_df['output{}'.format(i)] = flat_data
            else:
                self.dtypes.append('REAL')
                output_df['output{}'.format(i)] = out_data

        # create table if it doesn't yet exist; if it does, check the schema matches the new data
        exists_query = 'SELECT name FROM sqlite_master WHERE type="table" AND name="{}";'
        exists_query = exists_query.format(self.table_name)
        if self.db.execute(exists_query).fetchone():
            self._check_schema(output_df)
        else:
            cols = ', '.join(['"output{}" {}'.format(i, dtype)
                              for i, dtype in enumerate(self.dtypes)])
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

    def read(self, cols=None, rows=None):
        """Retrieve all processed features from cache, or lookup a single observation.

        For features that are multi-dimensional, use pickle to read string.

        Parameters
        ----------
        cols : list of int (default: None)
            indices of output columns to retrieve. If None, get all columns.
        rows: list of any or any (default: None)
            index value to lookup output for, in dictionary form. If None, get all rows.
            should be the same data type as the index.
        """
        if cols:
            cols = ['output{}'.format(c) for c in utils.generic.listify(cols)]
            cols = ', '.join(cols)
            query = "SELECT {}, ind FROM {} ".format(cols, self.table_name)
        else:
            query = "SELECT * FROM {} ".format(self.table_name)

        if rows:
            rows = utils.generic.listify(rows)
            query += "WHERE ind IN ({})".format(','.join(rows))

        out = list(pd.read_sql_query(query, self.db, index_col='ind').values.T)
        for i in range(len(out)):
            if self.dtypes[i] == 'TEXT':
                out[i] = np.array([pickle.loads(v) for v in out[i]])
                out[i] = out[i].reshape([-1] + list(self.original_shapes[i]))
        return out
