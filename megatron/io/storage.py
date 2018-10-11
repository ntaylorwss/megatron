import re
import sqlite3
import pandas as pd


class LocalStorage:
    def __init__(self, table_name):
        self.db = sqlite3.connect('megatron_default')
        if table_name:
            self.table_name = table_name
        else:
            tables = self.db.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()
            n_unnamed_tables = len([table for table in tables if table[0][:9]=='pipeline_'])
            self.table_name = 'pipeline_{0:03d}'.format(n_unnamed_tables+1)
        self.exists_query = 'SELECT name FROM sqlite_master WHERE type="table" AND name="{}";'
        self.exists_query = self.exists_query.format(self.table_name)
        self.make_query = 'CREATE TABLE {} ({}, UNIQUE ({}));'
        self.input_names = []

    def delete_table(self, table_name=None):
        if table_name is None:
            table_name = self.table_name
        self.db.execute('DROP TABLE IF EXISTS {}'.format(table_name))
        self.db.commit()

    def write(self, input_data, output_data):
        """Write set of observations to database, which includes input data and output data."""
        in_df = pd.DataFrame(input_data)
        in_df.columns = ['in_{}'.format(col) for col in in_df.columns]
        out_df = pd.DataFrame(output_data)
        out_df.columns = ['out_{}'.format(col) for col in out_df.columns]

        # drop duplicate observations since cache expects no duplicates
        in_df.drop_duplicates(inplace=True)
        out_df.drop_duplicates(inplace=True)

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
            # if table hasn't been used yet, create it
            cols = ', '.join(['`{}` blob'.format(name) for name in in_df.columns]) + ', '
            cols += ', '.join(['`{}` blob'.format(name) for name in out_df.columns])
            unique = ', '.join(in_df.columns)
            self.db.execute(self.make_query.format(self.table_name, cols, unique))
            self.db.commit()

        df = pd.concat([in_df, out_df], axis=1)
        df.to_sql(self.table_name, self.db, if_exists='append', index=False)
        self.input_names = list(input_data.keys())

    def lookup(self, output_cols=None, lookup_obs=None, **kwargs):
        out = pd.read_sql_table(self.table_name, self.db, **kwargs)
        if lookup_obs:
            masks = []
            for col in self.input_names:
                same_obs = out[col] == lookup_obs[col]
                while len(same_obs.shape) > 1:
                    same_obs = same_obs.all(axis=-1)
                masks.append(same_obs)
            mask = np.logical_and.reduce(masks)
            out = out.loc[mask]

        if output_cols:
            output_cols = ['out_{}'.format(col) for col in output_cols]
            out = out[output_cols]
            out.columns = [col[4:] for col in output_cols]
        else:
            # not robust
            output_cols = [col for col in out.columns if col[:4]=='out_']
            out = out[output_cols]
            out.columns = [col[4:] for col in output_cols]
        return out
