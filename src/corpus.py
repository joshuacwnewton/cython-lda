from numpy import array, ndarray
import pickle as pickle

from src.alphabet import Alphabet


class Document(object):

    def __init__(self, corpus, name, tokens):

        assert isinstance(corpus, Corpus)
        assert isinstance(name, str)
        assert isinstance(tokens, ndarray)

        self.corpus = corpus
        self.name = name
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def plaintext(self):
        return ' '.join([self.corpus.alphabet.lookup(x) for x in self.tokens])


class Corpus(object):

    def __init__(self):

        self.documents = []
        self.alphabet = Alphabet()

    def add(self, name, data):

        assert isinstance(name, str)
        assert isinstance(data, list)

        tokens = array([self.alphabet[x] for x in data])
        self.documents.append(Document(self, name, tokens))

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename, 'r'))

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
