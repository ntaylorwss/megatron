import unittest
import os
import sqlite3
import numpy as np
import pandas as pd
from megatron.io.generator import PandasGenerator, CSVGenerator, SQLGenerator


class test_PandasGenerator(unittest.TestCase):
    def test_invalid(self):
        # test error cases
        df = pd.DataFrame(np.zeros((100, 5)), columns=['a', 'b', 'c', 'd', 'e'])
        self.assertRaises(ValueError, PandasGenerator, dataframe=df, exclude_cols=['f'])
        self.assertRaises(ValueError, PandasGenerator, dataframe=df, batch_size=-1)
        self.assertRaises(ValueError, PandasGenerator, dataframe=df, batch_size=0)

    def test_zeros(self):
        data_gen = PandasGenerator(pd.DataFrame(np.zeros((1000, 2)), columns=['a', 'b']),
                                   batch_size=100)
        output = next(data_gen)

        # data type is correct
        assert isinstance(output, dict)

        # dict keys are correct
        correct_keys = set(['a', 'b'])
        output_keys = set(output)
        assert correct_keys == output_keys

        # dict values are correct
        correct_values = [np.zeros(100) for i in range(2)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

    def test_last_batch(self):
        data_gen = PandasGenerator(pd.DataFrame(np.zeros((1000, 2)), columns=['a', 'b']),
                                   batch_size=400)
        for i in range(2):
            output = next(data_gen)
        output = next(data_gen)

        # shape is correct
        correct_values = [np.zeros(200) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

    def test_reset(self):
        data_gen = PandasGenerator(pd.DataFrame(np.zeros((1000, 2)), columns=['a', 'b']),
                                   batch_size=400)
        for i in range(3):
            output = next(data_gen)
        output = next(data_gen)

        # data type is correct
        assert isinstance(output, dict)

        # dict values are correct
        correct_values = [np.zeros(400) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))


class test_CSVGenerator(unittest.TestCase):
    def setUp(self):
        pd.DataFrame(np.zeros((1000, 3)), columns=['a', 'b', 'c']).to_csv('test.csv', index=False)

    def tearDown(self):
        os.remove('test.csv')

    def test_invalid(self):
        self.assertRaises(FileNotFoundError, CSVGenerator, filepath='doesnt_exist.csv')

        self.assertRaises(ValueError, CSVGenerator, filepath='test.csv', exclude_cols=['d'])
        self.assertRaises(ValueError, CSVGenerator, filepath='test.csv', batch_size=-1)
        self.assertRaises(ValueError, CSVGenerator, filepath='test.csv', batch_size=0)

    def test_zeros(self):
        data_gen = CSVGenerator('test.csv', batch_size=100)
        output = next(data_gen)

        # data type is correct
        assert isinstance(output, dict)

        # dict keys are correct
        correct_keys = set(['a', 'b', 'c'])
        output_keys = set(output)
        assert correct_keys == output_keys

        # dict values are correct
        correct_values = [np.zeros(100) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

    def test_last_batch(self):
        data_gen = CSVGenerator('test.csv', batch_size=400)
        for i in range(2):
            output = next(data_gen)
        output = next(data_gen)

        # shape is correct
        correct_values = [np.zeros(200) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

    def test_reset(self):
        data_gen = CSVGenerator('test.csv', batch_size=400)
        for i in range(3):
            output = next(data_gen)
        output = next(data_gen)

        # data type is correct
        assert isinstance(output, dict)

        # dict values are correct
        correct_values = [np.zeros(400) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))


class test_SQLGenerator(unittest.TestCase):
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
        data_gen = SQLGenerator(self.conn, "SELECT * FROM test", batch_size=2)
        output = next(data_gen)

        # data type is correct
        assert isinstance(output, dict)

        # dict keys are correct
        correct_keys = set(['a', 'b', 'c'])
        output_keys = set(output)
        assert correct_keys == output_keys

        # dict values are correct
        correct_values = [np.zeros(2) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

    def test_last_batch(self):
        data_gen = SQLGenerator(self.conn, "SELECT * FROM test", batch_size=2)
        for i in range(2):
            output = next(data_gen)
        output = next(data_gen)

        # shape is correct
        correct_values = [np.zeros(1) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))

    def test_reset(self):
        data_gen = SQLGenerator(self.conn, "SELECT * FROM test", batch_size=2)
        for i in range(3):
            output = next(data_gen)
        output = next(data_gen)

        # data type is correct
        assert isinstance(output, dict)

        # dict values are correct
        correct_values = [np.zeros(400) for i in range(3)]
        output_values = list(output.values())
        assert all(np.array_equal(output, correct)
                   for output, correct in zip(output_values, correct_values))
