import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.utils import shuffle

from tensorflow.keras.models import Model 
import tensorflow_hub as hub

train_dataset_fp = './data/local_train_set.csv'
test_dataset_fp = './data/local_test_set.csv'

trainset = pd.read_csv(train_dataset_fp)
testset = pd.read_csv(test_dataset_fp)

from utility import BertTokenizer, CleanedTextDict

bert_layer = hub.KerasLayer("./bert_layer", trainable=True)

tokenizer = BertTokenizer(max_len = 256, bert_layer = bert_layer)
#textDic = CleanedTextDict(trainset, testset)

trainset['stokens'] = trainset['CleanedText'].apply(lambda x: tokenizer.GetStokens(x))
trainset['input_ids'] = trainset['stokens'].apply(lambda x: tokenizer.GetInput_ids(x))
trainset['input_masks'] = trainset['stokens'].apply(lambda x: tokenizer.GetInput_masks(x))
trainset['input_segments'] = trainset['stokens'].apply(lambda x: tokenizer.GetInput_segments(x))

X = [tf.convert_to_tensor(trainset['input_ids'], dtype=tf.int32),
    tf.convert_to_tensor(trainset['input_masks'], dtype=tf.int32),
    tf.convert_to_tensor(trainset['input_segments'], dtype=tf.int32),
]

def ScoreToTensor(raw_Y):
    Y = np.array(raw_Y) - 1 # convert to [0, 4]
    Y = [[int(t == label) for t in range(5)] for label in Y]
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    
    return Y

Y = ScoreToTensor(trainset['Score'])

max_len = 256

input_id = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
input_segment = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)

pooled_output, sequence_output = bert_layer([input_id, input_mask, input_segment])

F1 = keras.layers.Dense(64, activation='relu')(pooled_output)
F2 = keras.layers.Dropout(0.2)(F1)
F3 = keras.layers.Dense(5, activation='softmax')(F2)

model = Model(inputs=[input_id, input_mask, input_segment], outputs=F3)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

import datetime

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=X, y=Y, epochs = 1, validation_split = 0.2, shuffle='steps_per_epoch', callbacks=[tensorboard_callback])






'''
column_names = ['Product_ID', 'User_ID', \
    'Time_ID', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'CleanedText', 'Score']
label_name = 'Score'

batch_size = 2
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name)

features, labels = next(iter(train_dataset))

pack_features_vector(tokenizer, textDic)(features, labels)
'''
#train_dataset = train_dataset.map(pack_features_vector)