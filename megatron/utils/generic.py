import inspect
import functools
import collections


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


def listify(x):
    return x if isinstance(x, list) else [x]


def delistify(x):
    return x[0] if isinstance(x, list) else x


def isinstance_str(obj, classname):
    return classname in [t.__name__ for t in obj.__class__.__mro__[:-1]]


def flatten(L):
    for l in L:
        if isinstance(l, collections.Iterable) and not isinstance(l, (str, bytes)):
            yield from flatten(l)
        else:
            yield l


class IndexedData:
    def __init__(self, data, index):
        self.data = data
        self.index = index
