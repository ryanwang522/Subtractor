import numpy as np
from CharacterTable import *

def genData(quesSize=80000, digits=3):
    questions = []
    expected = []
    seen = set()

    print('Generating data...')

    while len(questions) < quesSize:
        func = lambda: int(''.join(np.random.choice(list('0123456789'))
                        for i in range(np.random.randint(1, digits + 1))))
        a, b = func(), func()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)

        e = '{}-{}'.format(a, b)
        expression = e + " " * ((2 * digits + 1) - len(e))
        ans = str(a - b)
        ans += " " * (digits + 1 - len(ans)) # `+1` for the carry

        questions.append(expression)
        expected.append(ans)
    
    print('Generating data complete.')
    print('Total questions:', len(questions))
    return questions, expected

def Vectorize(questions, expected, ctable, digits=3):
    print('Vectorization begin...')
    MAXLEN = 2 * digits + 1
    x = np.zeros((len(questions), MAXLEN, len(ctable.chars)), dtype=np.bool)
    y = np.zeros((len(expected), digits + 1, len(ctable.chars)), dtype=np.bool)

    # encoding the each char in expression/answer  to boolean value
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, digits + 1)
    
    print('Vectorization complete')
    return x, y

def splitData(x, y, splitAt=20000):
    indices = np.arange(len(y))
    np.random.shuffle(indices) # why shuffle ??
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x[:splitAt]
    train_y = y[:splitAt]
    test_x = x[splitAt:]
    test_y = y[splitAt:]

    # train validation split
    splitAt = len(train_x) - len(train_x) // 10
    (val_x, train_x) = train_x[splitAt:], train_x[:splitAt]
    (val_y, train_y) = train_y[splitAt:], train_y[:splitAt]

    return ((train_x, train_y), (val_x, val_y), (test_x, test_y))

def main():
    ctable = CharacterTable("0123456789- ")

    questions, expected = genData()
    x, y = Vectorize(questions, expected, ctable)
    
    ((train_x, train_y), (val_x, val_y), (test_x, test_y)) = splitData(x, y)

    """
    t = ""
    for expr in train_x:
        t += ctable.decode(expr)
        break
    t += " = "
    for expr in train_y:
        t += ctable.decode(expr)
        break
    print(t)
    """

if __name__ == "__main__":
    main()