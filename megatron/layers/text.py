import numpy as np
from .core import StatelessLayer
from ..layertools.wrappers import _vectorized_func

try:
    import nltk
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
    def __init__(self, language='english'):
        super().__init__(language=language)
        self.transform = np.vectorize(self._transform)
        nltk.download('stopwords', quiet=True)

    def _transform(self, X):
        stops = set(stopwords.words(self.kwargs['language']))
        return ' '.join(word for word in X.split(' ') if word not in stops)
