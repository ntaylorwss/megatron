import hashlib
import inspect
import functools
import numpy as np


def initializer(func):
    # https://stackoverflow.com/questions/1389180/automatically-initialize-instance-variables.
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kargs):
        parameters = [sig.parameters[k] for k in sig.parameters]
        for parameter, arg in zip(parameters[1:len(args)+1], args):
            setattr(self, parameter.name, arg)
        for parameter in parameters[len(args)+1:]:
            if parameter.name in kargs:
                setattr(self, parameter.name, kargs[parameter.name])
            else:
                setattr(self, parameter.name, parameter.default)
        func(self, *args, **kargs)

    return wrapper


class EagerRunException(Exception):
    def __init__(self):
        message = "Pipeline.run() should not be called when running in Eager Execution mode."
        super().__init__(message)


class ShapeError(Exception):
    def __init__(self, name, input_dims):
        msg = "Data fed into '{}' has {} dims; should be 1D array".format(name, input_dims)
        super().__init__(msg)


class PipelineError(Exception):
    pass


def listify(x):
    return x if isinstance(x, list) else [x]


def delistify(x):
    return x[0] if isinstance(x, list) else x


def md5_hash(x):
    if x.__class__.__name__ == 'ndarray':
        x = bytes(x)
    return str(int(hashlib.md5(str(x).encode()).hexdigest(), 16))


def column_stack(arrays):
    """Given a list of arrays, some of which are 2D, turn the 2D ones into 1D arrays."""
    out = []
    for array in arrays:
        if len(array.shape) == 1:
            out.append(array)
        elif len(array.shape) == 2:
            for column in list(array.T):
                out.append(column)
        else:
            raise ValueError("An array has more than 2 dimensions")
    return np.stack(out).T


def safe_divide(X1, X2, impute=0):
    impute_array = np.ones_like(X1) * impute
    return np.divide(X1.astype(np.float16), X2, out=impute_array.astype(np.float16), where=X2!=0)
