import numpy as np
from .core import StatelessLayer
from ..utils import initializer

try:
    from nltk.corpus import stopwords
except ImportError:
    pass


class RemoveStopwords(StatelessLayer):
    """Remove common, low-information words from all elements of text array.

    Parameters
    ----------
    language : str (default: english)
        the language in which the text is written.
    """
    def __init__(self, language='english', name=None):
        super().__init__(name, language=language)

    def transform(self, X):
        stops = set(stopwords.words(self.kwargs['language']))
        return ' '.join(word for word in X.split(' ') if word not in stops)

# function will not apply natively to an array of strings; must vectorize
RemoveStopwords.transform = np.vectorize(RemoveStopwords.transform)
