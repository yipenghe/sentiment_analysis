import tensorflow
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
import csv
from mittens import Mittens

"""
Wrapper for fine tuning glove with glassdoor data
"""

class Cooccurrence(CountVectorizer):
    """
    This is from: https://github.com/titipata/cooccurence
    Co-ocurrence matrix
    Convert collection of raw documents to word-word co-ocurrence matrix
    Parameters
    ----------
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    max_df: float in range [0, 1] or int, default=1.0
    min_df: float in range [0, 1] or int, default=1
    Example
    -------
    >> import Cooccurrence
    >> docs = ['this book is good',
               'this cat is good',
               'cat is good shit']
    >> model = Cooccurrence()
    >> Xc = model.fit_transform(docs)
    Check vocabulary by printing
    >> model.vocabulary_
    """

    def __init__(self, encoding='utf-8', ngram_range=(1, 1),
                 max_df=1.0, min_df=1, max_features=None,
                 stop_words=None, normalize=True, vocabulary=None):

        super(Cooccurrence, self).__init__(
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words=stop_words,
            vocabulary=vocabulary
        )

        self.normalize = normalize

    def fit_transform(self, raw_documents, y=None):
        """Fit cooccurrence matrix
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        Xc : Cooccurrence matrix
        """
        X = super(Cooccurrence, self).fit_transform(raw_documents)
        n_samples, n_features = X.shape

        Xc = (X.T * X)
        if self.normalize:
            g = sp.diags(1./Xc.diagonal())
            Xc = g * Xc
        else:
            Xc

        return Xc

def read_doc(doc_name="pros_from_collection", restrict=0):
    """
    doc_name: sentence file for fine tuning, one sentence per line
    restrict: integer, restrict the number of reviews used for training
    return docs: list of list of strings
    """
    docs = []
    with open(doc_name) as f:
      lines = f.readlines()
      if restrict:
        lines=lines[:restrict]
      for text in lines:
        docs.append(text)
    return docs

def simple_glove2dict(glove_filename = "glove.6B.50d.txt"):
    """
    code from https://github.com/roamanalytics/mittens
    glove_file :glove embedding file
    return embed: dictionary mapping word to vector
    """
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

def create_word_list(vocab_dict):
    """
    vocab_dict: a dictionary mapping word to their index
    return ordered_word_list: list of words in their index order
    """
    index2word = {}
    for word in vocab_dict:
      index2word[vocab_dict[word]] = word
    ordered_word_list = []
    for i in range(len(vocab_dict)):
        ordered_word_list.append(index2word[i])
    return ordered_word_list