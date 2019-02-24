import unittest
import numpy as np
from megatron.utils.errors import ShapeError
from megatron.layers.image import RGBtoGrey, RGBtoBinary, Downsample, Upsample


class test_RGBtoGrey(unittest.TestCase):
    def setUp(self):
        self.transformers = {k: RGBtoGrey(k) for k in ['lightness', 'average', 'luminosity']}
        self.X_good = np.ones((48, 48, 3)) * 128
        self.X_bad = np.ones((48, 48)) * 128

    def test_transform(self):
        self.assertRaises(ShapeError, RGBtoGrey, self.X_bad)

        outputs = {k: self.transformers[k].transform(self.X_good)
                   for k in ['lightness', 'average', 'luminosity']}
        correct_outputs = {
            'lightness': np.ones((48, 48)) * 128,
            'average': np.ones((48, 48)) * 128,
            'luminosity': np.ones((48, 48)) * 128
        }
        assert all(outputs[k].shape == correct_outputs[k].shape for k in outputs)
        assert all(np.isclose(outputs[k], correct_outputs[k]).all() for k in outputs)

        output = RGBtoGrey(keep_dim=True).transform(self.X_good)
        correct_output = np.ones((48, 48, 1)) * 128
        assert np.isclose(output, correct_output).all()


class test_RGBtoBinary(unittest.TestCase):
    def setUp(self):
        self.transformer = RGBtoBinary()
        self.X_good = np.random.random((48, 48, 3))
        self.X_bad = np.random.random((48, 48))
        for i in range(48):
            self.X_good[i, i, :] = 0
            self.X_bad[i, i] = 0

    def test_transform(self):
        self.assertRaises(ShapeError, self.transformer.transform, self.X_bad)

        output = self.transformer.transform(self.X_good)
        correct_output = np.eye(48)
        assert np.array_equal(output, correct_output)


class test_Downsample(unittest.TestCase):
    def setUp(self):
        self.transformer_2D = Downsample((40, 40))
        self.transformer_3D = Downsample((40, 40, 3))
        self.X_3D = np.random.random((48, 48, 3))
        self.X_2D = np.random.random((48, 48))
        self.X_1D = np.random.random(48)

    def test_transform(self):
        self.assertRaises(ShapeError, self.transformer_2D.transform, self.X_1D)
        self.assertRaises(ValueError, Downsample((50, 50)).transform, self.X_2D)
        self.assertRaises(ValueError, Downsample((40,)).transform, self.X_2D)

        output_shape_2D = self.transformer_2D.transform(self.X_2D).shape
        correct_output_shape_2D = (40, 40)
        assert output_shape_2D == correct_output_shape_2D

        output_shape_3D = self.transformer_3D.transform(self.X_3D).shape
        correct_output_shape_3D = (40, 40, 3)
        assert output_shape_3D == correct_output_shape_3D


class test_Upsample(unittest.TestCase):
    def setUp(self):
        self.transformer_2D = Upsample((50, 50))
        self.transformer_3D = Upsample((50, 50, 3))
        self.X_3D = np.random.random((48, 48, 3))
        self.X_2D = np.random.random((48, 48))
        self.X_1D = np.random.random(48)

    def test_transform(self):
        self.assertRaises(ShapeError, self.transformer.transform, self.X_1D)
        self.assertRaises(ValueError, Upsample((40, 40)).transform, self.X_2D)
        self.assertRaises(ValueError, Upsample((50,)).transform, self.X_2D)

        output_shape_2D = self.transformer_2D.transform(self.X_2D).shape
        correct_output_shape_2D = (50, 50)
        assert output_shape_2D == correct_output_shape_2D

        output_shape_3D = self.transformer_3D.transform(self.X_3D).shape
        correct_output_shape_3D = (50, 50, 3)
        assert output_shape_3D == correct_output_shape_3D
