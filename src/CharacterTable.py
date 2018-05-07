import numpy as np

class CharacterTable(object):
    def __init__(self, chars="0123456789- "):
        self.chars = sorted(set(chars))

        # build correspondence table between char and index of encoding
        self.charToIndex = dict((c, i) for i, c in enumerate(self.chars))
        self.indexToChar = dict((i ,c) for i, c in enumerate(self.chars))

    def encode(self, expr, rowsNum):
        x = np.zeros(shape=(rowsNum, len(self.chars)))
        for i, c in enumerate(expr):
            x[i, self.charToIndex[c]] = 1
        return x 

    def decode(self, x, argmax=True):
        if argmax:
            # get the index of max value(=1) of each row
            x = x.argmax(axis=1)
        # then look up indexToChar table to decode
        return "".join(self.indexToChar[i] for i in x)
