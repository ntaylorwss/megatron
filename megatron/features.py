from .core import Feature


class NumericFeature(Feature):
    def __init__(self, graph, name):
        super().__init__(graph, name, 1)

    def validate_input(self, X):
        if len(X.shape) != 1:
            msg = "{} has incorrect number of dimensions. Should be 1-dimensional array."
            raise ValueError(msg.format(self.name))


class FeatureSet(Feature):
    def __init__(self, graph, name, n_dims):
        super().__init__(graph, name, n_dims)

    def validate_input(self, X):
        if len(X.shape) != 2:
            msg = "{} has incorrect number of dimensions. Should be 2-dimensional array."
            raise ValueError(msg.format(self.name))
        if X.shape[1] != self.n_dims:
            msg = "{} has incorrect number of features, should have {}."
            raise ValueError(msg.format(self.name, self.n_dims))


class TextFeature(Feature):
    def __init__(self, graph, name):
        super().__init__(graph, name, 1)

    def validate_input(self, X):
        if X.dtype != np.dtype('<U3'):  # string datatype
            raise ValueError("{} does not have text data type (<U3 in Numpy).".format(self.name))


class ImageFeature(Feature):
    def __init__(self, graph, name, shape):
        super().__init__(graph, name, 2)
        self.shape = shape

    def validate_input(self, X):
        if len(X.shape) != 3:
            msg = "{} has incorrect number of dimensions. Should be 3-dimensional array."
            raise ValueError(msg.format(self.name))
        if X.shape[1:] != self.shape:
            msg = "{} has incorrect image shape, should be {}."
            raise ValueError(msg.format(self.name, self.shape))
