import numpy as np
import random
from keras.models import Sequential, load_model
from keras import layers
from CharacterTable import *
import argparse

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

def genData(quesSize=150000, digits=3):
    questions = []
    expected = []
    seen = set()

    print('Generating data...')

    func = lambda: int(''.join(np.random.choice(list('0123456789'))
                        for i in range(np.random.randint(1, digits + 1))))
    while len(questions) < quesSize:
        a, b = func(), func()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        
        if random.randint(0, 99) % 2 == 0:
            e = str(a).zfill(digits) + '-' + str(b).zfill(digits)
            ans = str(a - b)
        else:
            e = str(a).zfill(digits) + '+' + str(b).zfill(digits)
            ans = str(a + b).zfill(4)
        
        ans = "1" + ans[1:].zfill(4) if int(ans) < 0 else "0" + ans.zfill(4)

        questions.append(e)
        expected.append(ans)
    
    print('Generating data complete.')
    print('Total questions:', len(questions))
    return questions, expected

def Vectorize(questions, expected, ctable, digits=3):
    print('Vectorization begin...')
    MAXLEN = 2 * digits + 1
    x = np.zeros((len(questions), MAXLEN, len(ctable.chars)))
    y = np.zeros((len(expected), digits + 2, len(ctable.ansChars)))

    # encoding the each char in expression/answer to boolean value
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN, "expr")
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, digits + 2, "ans")
    
    print('Vectorization complete')
    return x, y

def splitData(x, y, splitAt=120000):

    # train_test_split
    train_x = x[:splitAt]
    train_y = y[:splitAt]
    test_x = x[splitAt:]
    test_y = y[splitAt:]

    # train validation split for RNN
    # splitAt = len(train_x) - len(train_x) // 10
    # (val_x, train_x) = train_x[splitAt:], train_x[:splitAt]
    # (val_y, train_y) = train_y[splitAt:], train_y[:splitAt]

    # return ((train_x, train_y), (val_x, val_y), (test_x, test_y))
    return ((train_x, train_y), (test_x, test_y))

def buildModel(train_x, train_y):
    print('Building model...')
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    digits = 3
    
    # flattern the data
    train_x = train_x.reshape(len(train_x), -1) # 7 * 12 -> 1 * 84
    print(train_x.shape)

    train_y_sign = train_y[:, 0, :2].reshape(len(train_y), -1)
    train_y_num = train_y[:, 1:len(train_y), :].reshape(len(train_y), -1)
    print(train_y_num.shape)
    print(train_y_sign.shape)
    
    filepath = "../model/model-num-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'"
    modelOfNums = Sequential()
    modelOfNums.add(layers.Dense(256 , input_shape=(84,), activation='relu'))
    modelOfNums.add(layers.Dense(200, activation='relu'))
    modelOfNums.add(layers.Dense(150, activation='relu'))
    modelOfNums.add(layers.Dense(100, activation='relu'))
    modelOfNums.add(layers.Dense(4*10, activation='sigmoid'))
    modelOfNums.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    modelOfNums.fit(train_x, train_y_num,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2, 
                    shuffle=True, verbose=1, epochs=100)

    
    modelOfSign = Sequential()
    modelOfSign.add(layers.Dense(256 , input_shape=(84,), activation='relu'))
    modelOfSign.add(layers.Dense(64, activation='relu')) 
    modelOfSign.add(layers.Dense(2, activation='sigmoid'))
    modelOfSign.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    modelOfSign.fit(train_x, train_y_sign,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2, 
                    shuffle=True, verbose=1, epochs=80)
    
    return modelOfNums, modelOfSign

def transform(ans):
    if ans[0] == '1':
        ans = "-" + ans[1:len(ans)]
    return int(ans)

def validation(modelOfNums, modelOfSign, test_x, test_y, ctable):
    print('Validation with testing data...')
    right = 0

    predNums = modelOfNums.predict(test_x.reshape(len(test_x), -1), verbose=0)
    predNums = predNums.reshape((len(test_y), 4, 10))
    predSign = np.argmax(modelOfSign.predict(test_x.reshape(len(test_x), -1), verbose=0), axis=1)

    for i in range(len(predSign)):
        q = ctable.decode(test_x[i], type="expr")
        correct = transform(ctable.decode(test_y[i], type="ans"))
        
        guessNum = ctable.decode(predNums[i], type="ans")
        if predSign[i] == 0:
            guessSign = "+"
        elif predSign[i] == 1:
            guessSign = "-"
        guess = int(guessSign + guessNum)
        
        if correct == guess:
                right += 1
        
        if i >= len(predSign) - 10:
            print(q, end=' ')
            print(correct, end='\t')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)

    print('Validation complete.')        
    print("Accuracy : {0:3.2f} %".format((right / len(predNums)) * 100), end=' ')
    print("({} / {})".format(right, len(predNums)))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain',
                        default=False,
                        help='input T/F to retrain model')
    args = parser.parse_args()

    # Generate data
    ctable = CharacterTable()
    questions, expected = genData()
    x, y = Vectorize(questions, expected, ctable)
    ((train_x, train_y), (test_x, test_y)) = splitData(x, y)
 
    if not args.retrain:
        # Use the best model trained on jupyter so far
        modelDir = "../model/"
        
        # modelOfNums = load_model(modelDir + "nums_model")
        # modelOfSign = load_model(modelDir + "sign_model")
        # validation(modelOfNums, modelOfSign, test_x, test_y, ctable)

        modelOfNumsRev = load_model(modelDir + "nums_model_revised")
        modelOfSignRev = load_model(modelDir + "sign_model_revised")
        validation(modelOfNumsRev, modelOfSignRev, test_x, test_y, ctable)

    else:
        # retrain the model by training data and output the accuracy result
        modelOfNums, modelOfSign = buildModel(train_x, train_y)
        validation(modelOfNums, modelOfSign, test_x, test_y, ctable)

if __name__ == "__main__":
    main()