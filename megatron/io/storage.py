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
    def __init__(self, table_name, db_conn=None):
        if db_conn:
            self.db = db_conn
        else:
            self.db = sqlite3.connect('megatron_default')
        if table_name:
            self.table_name = table_name
        else:
            tables = self.db.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()
            n_unnamed_tables = len([table for table in tables if table[0][:9]=='pipeline_'])
            self.table_name = 'pipeline_{0:03d}'.format(n_unnamed_tables+1)
        self.exists_query = 'SELECT name FROM sqlite_master WHERE type="table" AND name="{}";'
        self.exists_query = self.exists_query.format(self.table_name)
        self.make_query = 'CREATE TABLE {} (ind VARCHAR, {}, UNIQUE ({}));'
        self.input_names = []

    def delete_table(self):
        """Remove this table from the database."""
        self.db.execute('DROP TABLE IF EXISTS {}'.format(self.table_name))
        self.db.commit()

    def write(self, input_data, output_data, input_index):
        """Write set of observations to database.

        Parameters
        ----------
        input_data : dict of ndarray
            input fields and associated data fed through pipeline.
        output_data : dict of ndarray
            resulting features from applying pipeline to input_data.
        """
        in_df = pd.DataFrame(input_data)
        in_df.columns = ['in_{}'.format(col) for col in in_df.columns]
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

        # create table if it doesn't yet exist
        if self.db.execute(self.exists_query).fetchone():
            # if existing SQL colnames don't agree with this data, throw error that table is in use
            sql_cols = "SELECT sql FROM sqlite_master WHERE name='{}';"
            sql_cols = existing_cols_query.format(self.table_name)
            sql_cols = self.db.execute(sql_cols).fetchone()
            regex_query = r'`(?:in|out)_[^\s]+`'
            sql_cols = pd.Series([s[1:-1] for s in re.findall(regex_query, sql_cols)])
            data_cols = pd.Series(list(input_data) + list(output_data))
            if (sql_cols != data_cols).any():
                raise ValueError("Pipeline name already in use with different inputs/outputs.")
        else:
            cols = ', '.join(['`{}` blob'.format(name) for name in in_df.columns[1:]]) + ', '
            cols += ', '.join(['`{}` blob'.format(name) for name in out_df.columns])
            unique = ', '.join(in_df.columns[1:])
            self.db.execute(self.make_query.format(self.table_name, cols, unique))
            self.db.commit()

        # write data to db
        df = pd.concat([in_df, out_df], axis=1)
        df.to_sql(self.table_name, self.db, if_exists='append', index=False)
        self.db.execute("CREATE INDEX IF NOT EXISTS ind ON {} (ind)".format(self.table_name))
        self.db.commit()

        # save input and output field names for future reference
        self.input_names = list(input_data.keys())
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

        if lookup_vals:
            lookup_vals = utils.generic.listify(lookup_vals)
            query += "WHERE ind IN ({})".format(','.join(lookup_vals))

        return pd.read_sql_query(query, self.db, index_col='ind')
