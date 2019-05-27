import numpy as np
def sentence_rep(listOfWordVector, comb_func = np.mean):
    """
    Senetence representation stemming from word vectors
    listOfWordVector: a list of word vectors
    comb_func: a function that combines a list of word vectors into one single vector

    returns: a single vector representing the sentence
    """
    return comb_func(listOfWordVector)


def init_GloVe(glove_dim = 50):
    glove_src = os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(glove_dim))
    # Creates a dict mapping strings (words) to GloVe vectors:
    GLOVE = utils.glove2dict(glove_src)
    return GLOVE

def glove2dict(src_filename):
    """
    From CS224U github utils
    GloVe Reader.
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors.
    """
    data = {}
    with open(src_filename) as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def glove_vec(w):
    """Return `w`'s GloVe representation if available, else return
    a random vector.
    From CS224U github utils
    """
    return GLOVE.get(w, randvec(w, n=glove_dim))