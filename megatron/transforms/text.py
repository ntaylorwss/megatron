import numpy as np
try:
    from nltk.corpus import stopwords
except ImportError:
    pass


def remove_stopwords(X, language="english"):
    stops = set(stopwords.words("english"))
    return ' '.join(word for word in X.split(' ') if word not in stops)

# function will not apply natively to an array of strings; must vectorize
remove_stopwords = np.vectorize(remove_stopwords)



