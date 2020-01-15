import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.utils import shuffle

train_dataset_fp = './data/local_train_set.csv'
test_dataset_fp = './data/local_test_set.csv'

trainset = pd.read_csv(train_dataset_fp)
testset = pd.read_csv(test_dataset_fp)

from utility import BertTokenizer, CleanedTextDict

tokenizer = BertTokenizer(max_len = 256)
#textDic = CleanedTextDict(trainset, testset)

trainset['stokens'] = trainset['CleanedText'].apply(lambda x: tokenizer.GetStokens(x))
trainset['input_ids'] = trainset['stokens'].apply(lambda x: tokenizer.GetInput_ids(x))
trainset['input_masks'] = trainset['stokens'].apply(lambda x: tokenizer.GetInput_masks(x))
trainset['input_segments'] = trainset['stokens'].apply(lambda x: tokenizer.GetInput_segments(x))

X = [tf.convert_to_tensor(trainset['input_ids'], dtype=tf.int32),
    tf.convert_to_tensor(trainset['input_masks'], dtype=tf.int32),
    tf.convert_to_tensor(trainset['input_segments'], dtype=tf.int32),
]

Y = tf.convert_to_tensor(trainset['Score']-1, dtype=tf.float32)

from tensorflow.keras.models import Model 
import tensorflow_hub as hub

max_len = 256

input_id = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
input_segment = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)

bert_layer = hub.KerasLayer("./bert_layer", trainable=True)
pooled_output, sequence_output = bert_layer([input_id, input_mask, input_segment])

F1 = keras.layers.Dense(64, activation='relu')(pooled_output)
F2 = keras.layers.Dropout(0.2)(F1)
F3 = keras.layers.Dense(5, activation='softmax')(F2)

model = Model(inputs=[input_id, input_mask, input_segment], outputs=F3)
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x=X, y=Y, epochs = 3, validation_split = 0.2, shuffle='steps_per_epoch')

print(history)






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