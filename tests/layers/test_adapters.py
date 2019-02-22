import unittest
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from megatron.layers.adapters import Sklearn, Keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class test_Sklearn(unittest.TestCase):
    def setUp(self):
        self.transformer = Sklearn(StandardScaler())
        self.model = Sklearn(DecisionTreeClassifier())
        self.X = np.ones((100, 5))
        self.Y = np.zeros((100, 2)) + np.arange(2)

    def test_partial_fit(self):
        self.transformer.partial_fit(self.X)
        self.assertRaises(NotImplementedError, self.model.partial_fit, self.X, self.Y)

    def test_fit(self):
        self.transformer.fit(self.X)
        self.model.fit(self.X, self.Y)

    def test_transform(self):
        # shouldn't transform without fitting first
        self.assertRaises(NotFittedError, self.transformer.transform, self.X)
        self.transformer.fit(self.X)
        output = self.transformer.transform(self.X)
        correct_output = np.zeros(self.X.shape)
        assert np.array_equal(output, correct_output)

        self.assertRaises(NotFittedError, self.model.transform, self.X)
        self.model.fit(self.X, self.Y)
        output = self.model.transform(self.X, self.Y)
        correct_output = self.Y
        assert np.array_equal(output.argmax(axis=1), correct_output.argmax(axis=1))


class test_Keras(unittest.TestCase):
    def setUp(self):
        in_layer = Input((10,))
        out_layer = Dense(2)(in_layer)
        model = Model(in_layer, out_layer)
        model.compile(optimizer='adam', loss='mse')
        self.model = Keras(model)
        self.X = np.ones((100, 10))
        self.Y = np.zeros((100, 2)) + np.arange(2)

    def test_partial_fit(self):
        self.model.partial_fit(self.X, self.Y)

    def test_fit(self):
        self.model.fit(self.X, self.Y, verbose=0)

    def test_transform(self):
        self.model.fit(self.X, self.Y, epochs=100, verbose=0)
        output = self.model.transform(self.X)
        correct_output = self.Y
        assert np.array_equal(output.argmax(axis=1), correct_output.argmax(axis=1))
