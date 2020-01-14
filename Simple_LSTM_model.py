import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.utils import shuffle

def Build_Dict(trainset, testset):
    count_of_words = {}
    words = (' '.join(list(trainset['CleanedText'].values) + list(testset['CleanedText'].values))).split()
    
    for word in words:
        if not word in count_of_words:
            count_of_words[word] = 1
        else:
            count_of_words[word] += 1
    
    dic = {}
    for word in count_of_words:
        if count_of_words[word] <= 10: # Nan Threshold
            continue
        dic[word] = len(dic)
    
    return dic

def CleanedTextToTensor(raw_X):
    X = [x.split() for x in raw_X]
    X = [[dic[t] for t in x if t in dic] for x in X]
    
    max_len = max([len(x) for x in X])
    X = [x + [0] * (max_len - len(x)) for x in X]
    X = np.array(X)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    return X

def ScoreToTensor(raw_Y):
    Y = np.array(raw_Y) - 1 # convert to [0, 4]
    Y = [[int(t == label) for t in range(5)] for label in Y]
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    
    return Y

trainset = pd.read_csv("./data/local_train_set.csv")
testset = pd.read_csv("./data/local_test_set.csv")

dic = Build_Dict(trainset, testset)
Vocab_Size = len(dic) # Nan

print("Total vocabularies : %d" % Vocab_Size)

train_X = CleanedTextToTensor(trainset['CleanedText'].values)
print('train_X shape:', train_X.shape)
train_Y = ScoreToTensor(trainset['Score'].values)
print('train_Y shape:', train_Y.shape)

test_X = CleanedTextToTensor(testset['CleanedText'])
print('train_X shape:', test_X.shape)
test_Y = ScoreToTensor(testset['Score'])
print('train_Y shape:', test_Y.shape)

hidden_size = 128

model = keras.Sequential([
    keras.layers.Embedding(Vocab_Size, hidden_size),
    keras.layers.Bidirectional(keras.layers.LSTM(hidden_size)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(x=train_X, y=train_Y, epochs = 3, validation_data=(test_X, test_Y), shuffle=
                    'steps_per_epoch')

preds = model.predict(test_X)
preds = preds.argmax(1)

truths = testset['Score'].values - 1

res = [[0]*5 for i in range(5)]

for pred, truth in zip(preds, truths):
    res[truth][pred] += 1

tot = truths.size // 5

for i in range(5):
    print([x / tot for x in res[i]])