import unittest
import os
import sqlite3
import numpy as np
import pandas as pd
from megatron.io.dataset import PandasData, CSVData, SQLData


class test_PandasData(unittest.TestCase):
    def test_invalid(self):
        # test error cases
        df = pd.DataFrame(np.zeros((100, 5)), columns=['a', 'b', 'c', 'd', 'e'])
        self.assertRaises(ValueError, PandasData, dataframe=df, exclude_cols=['f'])
        self.assertRaises(ValueError, PandasData, dataframe=df, nrows=-1)
        self.assertRaises(ValueError, PandasData, dataframe=df, nrows=0)

    def test_zeros(self):
        # check properties of valid case
        output = PandasData(pd.DataFrame(np.zeros((100, 5)), columns=['a', 'b', 'c', 'd', 'e']),
                            exclude_cols=['e'], nrows=50)

        # data type is correct
        assert type(output) == dict

        # dict keys are correct
        correct_keys = set(['a', 'b', 'c', 'd'])
        output_keys = set(output)
        assert correct_keys == output_keys

        # dict values are correct
        correct_values = [np.zeros(50) for i in range(5)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))


class test_CSVData(unittest.TestCase):
    def test_invalid(self):
        self.assertRaises(FileNotFoundError, CSVData, filepath='doesnt_exist.csv')

        pd.DataFrame(np.zeros((100, 3)), columns=['a', 'b', 'c']).to_csv('test.csv', index=False)
        self.assertRaises(ValueError, CSVData, filepath='test.csv', exclude_cols=['d'])
        self.assertRaises(ValueError, CSVData, filepath='test.csv', nrows=-1)
        self.assertRaises(ValueError, CSVData, filepath='test.csv', nrows=0)
        os.remove('test.csv')

    def test_zeros(self):
        df = pd.DataFrame(np.zeros((100, 5)), columns=['a', 'b', 'c', 'd', 'e'])
        df.to_csv('test.csv', index=False)
        output = CSVData('test.csv', exclude_cols=['e'], nrows=50)

        # data type is correct
        assert type(output) == dict

        # dict keys are correct
        correct_keys = set(['a', 'b', 'c', 'd'])
        output_keys = set(output)
        assert correct_keys == output_keys

        # dict values are correct
        correct_values = [np.zeros(50) for i in range(5)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

        os.remove('test.csv')


class test_SQLData(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute('DROP TABLE IF EXISTS test;')
        self.conn.execute('''
            CREATE TABLE test (
                a INTEGER,
                b INTEGER,
                c INTEGER
            )
        ''')
        self.conn.execute('INSERT INTO test VALUES (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0);')

    def test_zeros(self):
        output = SQLData(self.conn, "SELECT * FROM test")

        # data type is correct
        assert type(output) == dict

        # dict keys are correct
        correct_keys = set(['a', 'b', 'c'])
        output_keys = set(output)
        assert correct_keys == output_keys

        # dict values are correct
        correct_values = [np.zeros(5) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))
