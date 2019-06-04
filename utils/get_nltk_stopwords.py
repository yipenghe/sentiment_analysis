import nltk
from nltk.corpus import stopwords
import re
def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  from https://github.com/harvardnlp/sent-conv-torch/blob/master/preprocess.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " ( ", string)
  string = re.sub(r"\)", " ) ", string)
  string = re.sub(r"\?", " ? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()

a = set(stopwords.words('english'))
total_stop_word = []
for word in a:
    total_stop_word.extend(clean_str(word).split(" "))
for word in set(total_stop_word):
    print(word)
