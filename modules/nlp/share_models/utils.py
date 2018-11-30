import re
from nltk.corpus import stopwords
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer

tokenize = WhitespaceTokenizer()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = GOOD_SYMBOLS_RE.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS])
    return text.strip()


def read_corpus(filename):
    data = []
    for line in open(filename, encoding='utf-8'):
        data.append(line.strip().split('\t'))
    return data


def prepare_file(in_, out_):
    out = open(out_, 'w')
    for line in open(in_, encoding='utf8'):
        line = line.strip().split('\t')
        new_line = [text_prepare(q) for q in line]
        print(*new_line, sep='\t', file=out)
    out.close()


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    global tokenize
    question = tokenize.tokenize(question)
    words_embeddings = [embeddings[word] for word in question if word in embeddings]
    if not words_embeddings:
        return np.zeros(dim)
    return np.mean(np.array(words_embeddings), axis=0)


#     # remove this when you're done
#     raise NotImplementedError(
#         "Open utils.py and fill with your code. In case of Google Colab, download"
#         "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
#         "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    embeddings = dict()
    dataFrame = pd.read_csv(embeddings_path, sep='\t')
    for key, value in dataFrame.iterrows():
        embeddings[value[0]] = np.array(value[1:])
    return embeddings, dataFrame.shape[1]-1
