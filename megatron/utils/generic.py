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


def flattenjson(json, delim='_'):
    def _flatten_dict(d):
        cols = {}
        for k in d.keys():
            if isinstance(d[k], dict ):
                get = _flatten_dict(d[k])
                for kk in get.keys():
                    cols[ k + delim + kk ] = get[kk]
            else:
                cols[k] = d[k]
        return cols

    if isinstance(json, dict):
        return _flatten_dict(json)
    else:
        return list(map(_flatten_dict, json))
