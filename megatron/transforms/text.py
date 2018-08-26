import numpy as np
from ..core import Transformation
from ..utils import initializer

try:
    from nltk.corpus import stopwords
except ImportError:
    pass


class RemoveStopwords(Transformation):
    """Remove common, low-information words from all elements of text array.

    Parameters
    ----------
    language : str (default: english)
        the language in which the text is written.
    """
    @initializer
    def __init__(self, language='english'):
        super().__init__()

    def transform(self, X):
        stops = set(stopwords.words(self.language))
        return ' '.join(word for word in X.split(' ') if word not in stops)

# function will not apply natively to an array of strings; must vectorize
RemoveStopwords.transform = np.vectorize(RemoveStopwords.transform)
