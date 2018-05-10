import numpy as np
from CharacterTable import *
from keras.models import Sequential
from keras import layers

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

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
        ans += " " * (digits + 1 - len(ans)) # `+1` for the carry or the minus sign.

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

    # encoding the each char in expression/answer to boolean value
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

    print('Build model...')
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    LAYERS = 1
    EPOCHS = 1
    digits = 3

    model = Sequential()
    model.add(layers.LSTM(HIDDEN_SIZE, input_shape=(2 * digits + 1, len(ctable.chars))))
    model.add(layers.RepeatVector(digits + 1))
    for _ in range(LAYERS):
        model.add(layers.LSTM(HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(ctable.chars))))
    model.add(layers.Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()

    for iteration in range(100):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(train_x, train_y,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(val_x, val_y))
        for i in range(10):
            ind = np.random.randint(0, len(val_x))
            rowx, rowy = val_x[np.array([ind])], val_y[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], argmax=False)
            print('Q', q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)    
    
    print('Validation with testing data...')
    right = 0
    preds = model.predict_classes(test_x, verbose=0)
    
    for i in range(len(preds)):
        q = ctable.decode(test_x[i])
        correct = ctable.decode(test_y[i])
        guess = ctable.decode(preds[i], argmax=False)
        if correct == guess:
            right += 1
        
    print("MSG : Accuracy is {}".format(right / len(preds)))


if __name__ == "__main__":
    main()