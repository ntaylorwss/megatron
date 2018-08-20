from ..core import Transformer


class Whiten(Transformer):
    def fit(self, X):
        #self.metadata['mean'] = X.mean(axis=0)
        self.metadata['sd'] = X.std(axis=0)

    def transform(self, X):
        return (X - self.metadata['mean']) / self.metadata['sd']


class Add(Transformer):
    def transform(self, X):
        return X + self.kwargs['add_this']


class Multiply(Transformer):
    def transform(self, X):
        return self.kwargs['factor'] * X


class Dot(Transformer):
    def transform(self, X):
        return np.dot(self.kwargs['W'], X)


class AddDim(Transformer):
    def transform(self, X):
        return np.expand_dims(X, self.kwargs['axis'])


class OneHot(Transformer):
    def transform(self, X):
        if not self.kwargs['max_val']:
            self.kwargs['max_val'] = X.max() + 1
        return (np.arange(self.kwargs['min_val'], self.kwargs['max_val']) == X[..., None]) * 1


class Reshape(Transformer):
    def transform(self, X):
        return np.reshape(X, self.kwargs['new_shape'])
