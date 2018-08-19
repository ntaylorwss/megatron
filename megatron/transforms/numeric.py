def whiten(X):
    return (X - X.mean(axis=-1)) / X.std(axis=-1)


def add(X, add_this):
    return X + add_this


def multiply(X, factor):
    return factor * X


def dot(X, W):
    return np.dot(W, X)


def add_dim(X, axis=-1):
    return np.expand_dims(X, axis)


def one_hot(X, max_val=None, min_val=0):
    if not max_val:
        max_val = X.max() + 1
    return (np.arange(min_val, max_val) == X[..., None]) * 1


def reshape(X, new_shape=None):
    return np.reshape(X, new_shape)
