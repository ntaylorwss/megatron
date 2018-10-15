import re
import sqlite3
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
        self.exists_query = 'SELECT name FROM sqlite_master WHERE type="table" AND name="{}";'
        self.exists_query = self.exists_query.format(self.table_name)
        self.make_query = 'CREATE TABLE {} (ind VARCHAR, {}, UNIQUE ({}));'
        self.output_names = []

    def _check_schema(self, input_data, output_data):
        """If existing SQL colnames and data colnames are off, throw error that table is in use.

        Parameters
        ----------
        input_data : dict of ndarray
            input fields and associated data fed through pipeline.
        output_data : dict of ndarray
            resulting features from applying pipeline to input_data.
        """
        sql_cols = "SELECT sql FROM sqlite_master WHERE name='{}';"
        sql_cols = sql_cols.format(self.table_name)
        sql_cols = self.db.execute(sql_cols).fetchone()[0]
        regex_query = r'"(?:in|out)_[^\s]+"'
        sql_cols = pd.Series([s[s.find('_')+1:-1] for s in re.findall(regex_query, sql_cols)])
        data_cols = pd.Series(list(input_data) + list(output_data))
        if (sql_cols != data_cols).any():
            raise ValueError("Pipeline name already in use with different inputs/outputs.")

    def write(self, input_data, output_data, input_index):
        """Write set of observations to database.

        Parameters
        ----------
        input_data : dict of ndarray
            input fields and associated data fed through pipeline.
        output_data : dict of ndarray
            resulting features from applying pipeline to input_data.
        """
        for k, v in input_data.items():
            input_data[k] = pd.Series(v, dtype=object)
        in_df = pd.DataFrame(input_data)
        in_df.columns = ['in_{}'.format(col) for col in in_df.columns]

        for k, v in output_data.items():
            output_data[k] = pd.Series(v, dtype=object)
        out_df = pd.DataFrame(output_data)
        out_df.columns = ['out_{}'.format(col) for col in out_df.columns]

        # apply index to input
        in_df.index = input_index

        # drop duplicate observations since cache expects no duplicates
        in_df.drop_duplicates(inplace=True)
        out_df.drop_duplicates(inplace=True)

        # put input index in column
        in_df.reset_index(inplace=True)
        in_df.rename({'index': 'ind'}, axis=1, inplace=True)
        in_df['ind'] = in_df.ind.astype(str)

        # create table if it doesn't yet exist; if it does, check the schema matches the new data
        if self.db.execute(self.exists_query).fetchone():
            self._check_schema(input_data, output_data)
        else:
            cols = ', '.join(['"{}" blob'.format(name) for name in in_df.columns[1:]]) + ', '
            cols += ', '.join(['"{}" blob'.format(name) for name in out_df.columns])
            unique = ', '.join(in_df.columns[1:])
            self.db.execute(self.make_query.format(self.table_name, cols, unique))
            self.db.commit()

        # delete data for indices that are being written
        inds = ','.join([str(i) for i in input_index.tolist()])
        self.db.execute("DELETE FROM {} WHERE ind IN ({})".format(self.table_name, inds))

        # write new data to db
        df = pd.concat([in_df, out_df], axis=1)
        df.to_sql(self.table_name, self.db, if_exists='append', index=False)
        self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS ind ON {} (ind)".format(self.table_name))
        self.db.commit()

        # save output field names for future reference
        self.output_names = list(output_data.keys())

    def read(self, output_cols=None, lookup=None):
        """Retrieve all processed features from cache, or lookup a single observation.

        Parameters
        ----------
        output_cols : list of str
            names of output columns to retrieve. If none, get all outputs.
        lookup_obs : dict of ndarray
            input data to lookup output for, in dictionary form.
        """
        if output_cols:
            output_cols = utils.generic.listify(output_cols)
            output_cols = ','.join(['out_{}'.format(c) for c in output_cols])
        else:
            output_cols = ','.join(['out_{}'.format(c) for c in self.output_names])
        query = "SELECT {}, ind FROM {} ".format(output_cols, self.table_name)

        if lookup:
            lookup = utils.generic.listify(lookup)
            query += "WHERE ind IN ({})".format(','.join(lookup))

        out = pd.read_sql_query(query, self.db, index_col='ind')
        n_rows = out.shape[0]
        out = dict(zip([col[4:] for col in out.columns], out.values.T))
        if n_rows == 1:
            out = {k: v[0] for k, v in out.items()}
        return out
