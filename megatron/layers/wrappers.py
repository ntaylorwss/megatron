from .core import Layer


class Sklearn(Layer):
    def __init__(self, sklearn_transformation):
        super().__init__()
        self.transformation = sklearn_transformation
        self.name = self.transformation.__class__.__name__

    @property
    def metadata(self):
        return self.transformation.__dict__

    def partial_fit(self, *inputs):
        self.transformation.partial_fit(*inputs)

    def fit(self, *inputs):
        self.transformation.fit(*inputs)

    def transform(self, *inputs):
        if hasattr(self.transformation, 'transform'):
            return self.transformation.transform(*inputs)
        else:
            # don't use the labels for this
            return self.transformation.predict(inputs[0])


class Keras(Layer):
    def __init__(self, keras_model):
        super().__init__()
        self.model = keras_model
        self.name = 'KerasModel'
        self.n_inputs = len(self.model.inputs)

    def partial_fit(self, *inputs):
        if self.n_inputs == 1:
            X, Y = inputs
        else:
            X = list(inputs[:self.n_inputs])
            Y = list(inputs[self.n_inputs:])
        self.model.fit(X, Y)

    def fit(self, *inputs):
        self.partial_fit(*inputs)

    def transform(self, *inputs):
        # don't use the labels for this
        if self.n_inputs == 1:
            return self.model.predict(inputs[0])
        else:
            return self.model.predict(list(inputs[:self.n_inputs]))
