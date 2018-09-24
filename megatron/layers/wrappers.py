from .core import StatelessLayer, StatefulLayer
from ..utils import md5_hash, initializer


class Lambda(StatelessLayer):
    """A layer holding a stateless transformation.

    For custom functions that are stateless, and thus do not require to be fit,
    a Lambda wrapper is preferred to creating a Transformation subclass.

    Parameters
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    **kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.

    Attributes
    ----------
    transform_fn : function
        the function to be applied, which accepts one or more
        Numpy arrays as positional arguments.
    **kwargs
        keyword arguments to whatever custom function is passed in as transform_fn.
    """
    def __init__(self, transform_fn, name=None, **kwargs):
        self.name = name if name else self.transform_fn.__name__
        self.transform_fn = transform_fn
        self.kwargs = kwargs

    def __str__(self):
        """Used in caching pipelines."""
        out = [str(kwarg) for kwarg in self.kwargs.values()]
        out.append(inspect.getsource(self.transform_fn))
        return ''.join(out)

    def transform(self, *inputs):
        """Applies transform_fn to given input data.

        Parameters
        ----------
        inputs : np.ndarray(s)
            input data to be passed to transform_fn; could be one array or a list of arrays.
        """
        return self.transform_fn(*inputs, **self.kwargs)


class SklearnTransformation(StatefulLayer):
    """Wrapper for Sklearn pipeline transformation classes.

    Parameters
    ----------
    transformation : sklearn.BaseEstimator
        the sklearn class to use.
    name : str
        name to give the transformation, used in visualization.

    Attributes
    ----------
    transformation : sklearn.BaseEstimator
        the sklearn class to use.
    name : str
        name to give the transformation, used in visualization.
    """
    @initializer
    def __init__(self, transformation, name=None):
        self.name = name if name else transformation.__class__.__name__

    def __str__(self):
        # when there's no metadata, string will be empty, which is like a unique null hash
        metadata = {k: v for k, v in self.transformation.__dict__.items() if k[-1] == '_'}
        metadata = ''.join([md5_hash(metadata) for metadata in metadata.values()])
        return '{}{}'.format(self.transformation.__class__.__name__, metadata)

    def fit(self, inputs):
        self.transformation.fit(inputs)

    def transform(self, inputs):
        return self.transformation.transform(inputs)
