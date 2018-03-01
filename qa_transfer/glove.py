import os
import zipfile
import pandas as pd
import csv

"""
https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
"""

class Glove(object):
    def __init__(self, pre_trained = True):
        self.GLOVE_PATH = os.getcwd() + '/data/glove.6B.zip'
        self.GLOVE_FILE = 'glove.6B.300d.txt' # can change it to glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt
        if pre_trained:
            glove = zipfile.ZipFile(self.GLOVE_PATH, 'r')
            words = pd.read_table(self.glove.open(GLOVE_FILE), sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            self.model = words.as_matrix()
        else:
            self.model = None

        if self.model:
            self.dict = {word: i for i, word in enumerate(words.index)}
        else:
            self.dict = {}

    def vectorize(self, word):
        if word in self.dict:
            index = self.dict[word]
            return self.model[index]
        else:
            return None

if __name__ == '__main__':
    model = Glove()
    print (model.vectorize('person'))
    print (model.vectorize('CDS'))
