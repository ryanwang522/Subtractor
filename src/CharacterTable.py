import numpy as np

class CharacterTable(object):
    def __init__(self, chars="0123456789+-"):
        self.chars = sorted(set(chars))
        self.ansChars = sorted(set('0123456789'))
        
        # build correspondence table between char and index of encoding
        self.charToIndex = dict((c, i) for i, c in enumerate(self.chars))
        self.indexToChar = dict((i ,c) for i, c in enumerate(self.chars))
        self.ansCharsToIndex =  dict((c, i) for i, c in enumerate(self.ansChars))
        self.indexToAnsChars = dict((i ,c) for i, c in enumerate(self.ansChars))

    def encode(self, expr, rowsNum, type):
        if type == "expr":
            x = np.zeros(shape=(rowsNum, len(self.chars)))
            for i, c in enumerate(expr):
                x[i, self.charToIndex[c]] = 1
        elif type == "ans":
            x = np.zeros(shape=(rowsNum, len(self.ansChars)))
            for i, c in enumerate(expr):
                x[i, self.ansCharsToIndex[c]] = 1

        return x 

    def decode(self, x, type, argmax=True):
        if argmax:
            # get the index of max value(=1) of each row
            x = x.argmax(axis=1)
        # then look up table to decode
        if type == "expr":
            return "".join(self.indexToChar[i] for i in x)
        elif type == "ans":
            return "".join(self.indexToAnsChars[i] for i in x)
