import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam



def DatasetToTensor(df):
    '''
    X = trainset['input_ids']
    print(X.shape)
    return 1,1
    '''
    
    convert = lambda s : (list(map(int, s[1:-1].split(','))))
    X = [tf.convert_to_tensor(trainset['Normalized_Product_ID'], dtype=tf.int32),
        tf.convert_to_tensor(trainset['Normalized_User_ID'], dtype=tf.int32),
        tf.convert_to_tensor(trainset['Normalized_Time_ID'], dtype=tf.int32),
        tf.convert_to_tensor(trainset['input_ids'].apply(convert), dtype=tf.int32),
        tf.convert_to_tensor(trainset['input_masks'].apply(convert), dtype=tf.int32),
        tf.convert_to_tensor(trainset['input_segments'].apply(convert), dtype=tf.int32),
    ]
    Y = [tf.convert_to_tensor(trainset['Score'] - 1, dtype=tf.int32),
        tf.convert_to_tensor(trainset['NormalizedHelpfulness'] - 1, dtype=tf.int32),
    ]

    return X, Y


def MultiModalModel():
    max_len = 128
    vocab_size = 1000
    word_dim = 64

    input_id = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ID')
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name = 'input_mask')
    input_segment = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name = 'input_seqment')

    bert_layer = hub.KerasLayer("../bert_layer", trainable=True)
    pooled_output, _ = bert_layer([input_id, input_mask, input_segment])

    input1 = keras.Input(shape=(1,), dtype=tf.float32, name='Product_ID')
    e1 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input1)

    input2 = keras.Input(shape=(1,), dtype=tf.float32, name='User_ID')
    e2 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input2)

    input3 = keras.Input(shape=(1,), dtype=tf.float32, name='Time_ID')
    e3 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input3)
    
    merge1 = keras.layers.Flatten()(keras.layers.concatenate([e1, e2, e3]))
    merge2 = keras.layers.concatenate([merge1, pooled_output])

    a1 = keras.layers.Dense(64, activation='relu')(merge2)
    a2 = keras.layers.Dropout(0.2)(a1)
    a3 = keras.layers.Dense(5, activation='softmax', name='Score')(a2)

    b1 = keras.layers.Dense(64, activation='relu')(merge2)
    b2 = keras.layers.Dropout(0.2)(b1)
    b3 = keras.layers.Dense(5, activation='softmax', name = 'Helpfulness')(b2)

    opt = Adam(lr=1e-5)
    model = keras.Model(inputs=[input1, input2, input3, input_id, input_mask, input_segment], \
        outputs=[a3, b3])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model

if __name__ == '__main__':
    trainset = pd.read_csv('./data/local_train_set.csv')
    testset = pd.read_csv('./data/local_test_set.csv')

    train_X, train_Y = DatasetToTensor(trainset)
    test_X, test_Y = DatasetToTensor(testset)

    model = MultiModalModel()

    checkpoints_dir = './checkpoints/'
    load_file = 'bert_model.h5'

    if load_file:
        model.load_weights(checkpoints_dir+load_file)

    history = model.fit(x=train_X, y=train_Y, epochs = 1, validation_data = (test_X, test_Y), shuffle='steps_per_epoch')

    

    model.save_weights(checkpoints_dir + 'bert_model.h5')







