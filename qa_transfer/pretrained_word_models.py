import os
import zipfile
import pandas as pd
import csv
import gensim


class Glove(object):
    """Pretrained Glove vectorizer
    Attributes:
        GLOVE_PATH  : path to the zipped file
        GLOVE_FILE  : file name. Can be selected from "glove.6B.50d.txt" "glove.6B.100d.txt" "glove.6B.200d.txt" or
                      "glove.6B.300d.txt"
        model       : 400000 x k matrix (k = number of embedding)
        dict        : dictionary that maps vocabulary to the index
    """
    def __init__(self, pre_trained=True):
        self.GLOVE_PATH = os.getcwd() + '/data/glove.6B.zip'
        self.GLOVE_FILE = 'glove.6B.300d.txt'  # can change it to glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt
        # load the file
        if pre_trained:
            glove = zipfile.ZipFile(self.GLOVE_PATH, 'r')
            words = pd.read_table(glove.open(self.GLOVE_FILE), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            self.model = words.as_matrix()
        else:
            self.model = None
        # build dictionary
        if self.model is not None:
            self.dict = {word: i for i, word in enumerate(words.index)}
        else:
            self.dict = {}

    def vectorize(self, word):
        """return vectorized word. If the word does not exist, return None
        Args:
            word: string to vectorize
        Returns:
            array with the length of k (k = number of embedding dimension), None otherwise.
        """
        if word in self.dict:
            index = self.dict[word]
            return self.model[index]
        else:
            return None

class Pretrained_Word2Vec(object):
    """Pretrained word2vec vectorizer
    Attributes:
        WORD2VEC_PATH  : path to the zipped file
        model          : gensim.models.keyedvectors.KeyedVectors
    """
    def __init__(self):
        self.WORD2VEC_PATH = os.getcwd() + '/data/GoogleNews-vectors-negative300.bin'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.WORD2VEC_PATH , binary=True)

    def vectorize(self, word):
        """return vectorized word. If the word does not exist, return None
        Args:
            word: string to vectorize
        Returns:
            array with the length of 300, None otherwise.
        """
        if word in self.model.wv.vocab.keys():
            return self.model[word]
        else:
            return None


if __name__ == '__main__':
    # model = Glove()
    model = Pretrained_Word2Vec()
    print (model.vectorize('person'))
    print (model.vectorize('CDS'))
