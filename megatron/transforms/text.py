import numpy as np
from ..core import Transformer
try:
    from nltk.corpus import stopwords
except ImportError:
    pass


class RemoveStopwords(Transformer):
    def transform(self, X):
        stops = set(stopwords.words(self.kwargs['language']))
        return ' '.join(word for word in X.split(' ') if word not in stops)

# function will not apply natively to an array of strings; must vectorize
RemoveStopwords.transform = np.vectorize(RemoveStopwords.transform)
