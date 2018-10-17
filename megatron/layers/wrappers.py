from .core import Layer


class SklearnLayer(Layer):
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
            return self.transformation.predict(*inputs)
