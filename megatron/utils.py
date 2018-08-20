import hashlib


class MetadataDict(dict):
    class MetadataKeyError(Exception):
        def __init__(self, key):
            message = ("Transforming call attempted to access metadata '{}' that "
                       "was not calculated in fit()")
            super().__init__(message.format(key))

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise MetadataDict.MetadataKeyError(key)


class KwargsDict(dict):
    class KwargKeyError(Exception):
        def __init__(self, key):
            message = ("Transforming call attempted to access keyword argument '{}' that "
                       "was not passed to Transformation initialization")
            super().__init__(message.format(key))

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise KwargsDict.KwargKeyError(key)


def catch_kwarg_error(fun):
    def wrapper(*args):
        try:
            return fun(*args)
        except KeyError as e:
            raise KeyError()
    return wrapper


class EagerRunException(Exception):
    def __init__(self):
        message = "Graph.run() should not be called when running in Eager Execution mode."
        super().__init__(message)


def listify(x):
    return x if isinstance(x, list) else [x]


def md5_hash(x):
    if x.__class__.__name__ == 'ndarray':
        x = bytes(x)
    return str(int(hashlib.md5(str(x).encode()).hexdigest(), 16))
