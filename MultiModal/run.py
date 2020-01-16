import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam

import argparse
from sklearn.metrics import f1_score


def DatasetToTensor(df, model):
    convert = lambda s : (list(map(int, s[1:-1].split(','))))
    max_len = 128

    if model == 'BERTMulModMulTask' or model == 'BERTMulMod':
        X = [tf.convert_to_tensor(df['Normalized_Product_ID'], dtype=tf.int32),
            tf.convert_to_tensor(df['Normalized_User_ID'], dtype=tf.int32),
            tf.convert_to_tensor(df['Normalized_Time_ID'], dtype=tf.int32),
            tf.convert_to_tensor(df['input_ids'].apply(convert), dtype=tf.int32),
            tf.convert_to_tensor(df['input_masks'].apply(convert), dtype=tf.int32),
            tf.convert_to_tensor(df['input_segments'].apply(convert), dtype=tf.int32),
        ]
    elif model == 'SingleLSTM':
        X = df['TextID'].apply(convert).values
        X = [x[:max_len] for x in X]

        max_len = max([len(x) for x in X])
        X = [x + [4500] * (max_len - len(x)) for x in X]
        X = np.array(X)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
    elif model == 'SingleBERT':
        X = [tf.convert_to_tensor(df['input_ids'].apply(convert), dtype=tf.int32),
            tf.convert_to_tensor(df['input_masks'].apply(convert), dtype=tf.int32),
            tf.convert_to_tensor(df['input_segments'].apply(convert), dtype=tf.int32),
        ]

    Y = [tf.convert_to_tensor(df['Score'] - 1, dtype=tf.int32),
        tf.convert_to_tensor(df['NormalizedHelpfulness'] - 1, dtype=tf.int32),
    ]

    return X, Y


def BERTMulModMulTask(lr):
    max_len = 128
    vocab_size = 1000
    word_dim = 32

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

    a3 = keras.layers.Dense(5, activation='softmax', name='Score')(merge2)
    b3 = keras.layers.Dense(5, activation='softmax', name = 'Helpfulness')(merge2)

    opt = Adam(lr=lr)
    model = keras.Model(inputs=[input1, input2, input3, input_id, input_mask, input_segment], \
        outputs=[a3, b3])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


def BERTMulMod(lr):
    max_len = 128
    vocab_size = 1000
    word_dim = 32

    input_id = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ID')
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name = 'input_mask')
    input_segment = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name = 'input_seqment')

    bert_layer1 = hub.KerasLayer("../bert_layer", trainable=True)
    bert_layer2 = hub.KerasLayer("../bert_layer", trainable=True)
    pooled_output1, _ = bert_layer1([input_id, input_mask, input_segment])
    pooled_output2, _ = bert_layer2([input_id, input_mask, input_segment])

    input1 = keras.Input(shape=(1,), dtype=tf.float32, name='Product_ID')
    e1 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input1)
    p1 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input1)

    input2 = keras.Input(shape=(1,), dtype=tf.float32, name='User_ID')
    e2 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input2)
    p2 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input2)

    input3 = keras.Input(shape=(1,), dtype=tf.float32, name='Time_ID')
    e3 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input3)
    p3 = keras.layers.Embedding(vocab_size, word_dim, embeddings_initializer=tf.random_normal_initializer)(
        input3)
    
    merge1 = keras.layers.Flatten()(keras.layers.concatenate([e1, e2, e3]))
    merge2 = keras.layers.concatenate([merge1, pooled_output1])

    merge3 = keras.layers.Flatten()(keras.layers.concatenate([p1, p2, p3]))
    merge4 = keras.layers.concatenate([merge3, pooled_output2])

    a3 = keras.layers.Dense(5, activation='softmax', name='Score')(merge2)
    b3 = keras.layers.Dense(5, activation='softmax', name = 'Helpfulness')(merge4)

    opt = Adam(lr=lr)
    model = keras.Model(inputs=[input1, input2, input3, input_id, input_mask, input_segment], \
        outputs=[a3, b3])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model

def SingleBERT(lr):
    max_len = 128
    vocab_size = 1000
    word_dim = 32

    input_id = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ID')
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name = 'input_mask')
    input_segment = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name = 'input_seqment')

    bert_layer1 = hub.KerasLayer("../bert_layer", trainable=True)
    bert_layer2 = hub.KerasLayer("../bert_layer", trainable=True)
    pooled_output1, _ = bert_layer1([input_id, input_mask, input_segment])
    pooled_output2, _ = bert_layer2([input_id, input_mask, input_segment])

    a3 = keras.layers.Dense(5, activation='softmax', name='Score')(pooled_output1)
    b3 = keras.layers.Dense(5, activation='softmax', name = 'Helpfulness')(pooled_output2)

    opt = Adam(lr=lr)
    model = keras.Model(inputs=[input_id, input_mask, input_segment], \
        outputs=[a3, b3])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


def SingleLSTM():
    Vocab_Size = 5000
    hidden_size = 128

    input1 = keras.Input(shape=(128, ), dtype=tf.float32, name='TextID')

    x1 = keras.layers.Embedding(Vocab_Size, hidden_size)(input1)
    x2 = keras.layers.Bidirectional(keras.layers.LSTM(hidden_size))(x1)
    x3 = keras.layers.Dropout(0.2)(x2)
    x4 = keras.layers.Dense(32, activation='relu')(x3)
    x5 = keras.layers.Dense(5, activation='softmax', name= 'Score')(x4)

    y1 = keras.layers.Embedding(Vocab_Size, hidden_size)(input1)
    y2 = keras.layers.Bidirectional(keras.layers.LSTM(hidden_size))(y1)
    y3 = keras.layers.Dropout(0.2)(y2)
    y4 = keras.layers.Dense(32, activation='relu')(y3)
    y5 = keras.layers.Dense(5, activation='softmax', name= 'Helpfulness')(y4)

    model = keras.Model(inputs=[input1], outputs=[x5, y5])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print("====================== Train Begin ===================")

        self.history = {}

        self.loss = []
        self.acc = []
        self.score = []
        self.help = []

        self.val_loss = []
        self.val_acc = []
        self.val_score = []
        self.val_help = []
    
    def on_train_end(self, logs={}):
        print("====================== Train End ===================")

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.score.append(logs.get('Score_accuracy'))
        self.help.append(logs.get('Helpfulness_accuracy'))

        if batch % 10 == 0:
            print("------------ Batch %d ---------" % batch)
            print(logs)
    
    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_score.append(logs.get('val_Score_accuracy'))
        self.val_help.append(logs.get('val_Helpfulness_accuracy'))

        print("============= Epoch %d ==========" % epoch)
        print(logs)

    def Output(self, file):
        print(self.loss, file=file)
        print(self.acc, file=file)
        print(self.score, file=file)
        print(self.help, file=file)
        print('\n', file=file)

        print(self.val_loss, file=file)
        print(self.val_acc, file=file)
        print(self.val_score, file=file)
        print(self.val_help, file=file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load', default=None
    )
    parser.add_argument(
        '--big', default=None
    )
    parser.add_argument(
        '--epoch', type=int
    )
    parser.add_argument(
        '--lr', type=float, default=0.00001
    )
    parser.add_argument(
        '--model', type=str,
            choices=['SingleLSTM', 'SingleBERT', 'BERTMulMod', 'BERTMulModMulTask']
    )
    args = parser.parse_args()
    print(args)

    if args.big != None:
        trainset = pd.read_csv('./data/train_set.csv')
        testset = pd.read_csv('./data/test_set.csv')
    else:
        trainset = pd.read_csv('./data/local_train_set.csv')
        testset = pd.read_csv('./data/local_test_set.csv')

    print("(trainset size : %d, testset size : %d)" % (len(trainset), len(testset)))

    train_X, train_Y = DatasetToTensor(trainset, args.model)
    test_X, test_Y = DatasetToTensor(testset, args.model)

    if args.model == 'BERTMulModMulTask':
        model = BERTMulModMulTask(args.lr)
    elif args.model == 'BERTMulMod':
        model = BERTMulMod(args.lr)
    elif args.model == 'SingleBERT':
        model = SingleBERT(args.lr)
    elif args.model == 'SingleLSTM':
        model = SingleLSTM()


    model_file = './checkpoints/' + args.model + '.h5'
    if args.load != None:
        model.load_weights(model_file)
    
    loss_history = LossHistory()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    save_model = keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')

    model.fit(x=train_X, y=train_Y, epochs = args.epoch, \
        validation_data = (test_X, test_Y), shuffle='steps_per_epoch', \
            callbacks=[early_stop, loss_history, save_model], verbose=0)

    logfp = open('log.' + args.model + '.txt', 'w')
    loss_history.Output(logfp)

    def Calc_F1(preds, truths, name):
        F1 = f1_score(truths, preds, average=None)
        print(name, ' F1 Score:  ', F1)
        print(name, ' F1 Score:  ', F1, file = logfp)

    score_preds, helpfulness_preds = model.predict(test_X)
    score_preds = score_preds.argmax(1) + 1
    helpfulness_preds = helpfulness_preds.argmax(1) + 1

    testset['score_preds'] = score_preds
    testset['helpfulness_preds'] = helpfulness_preds

    score_truths = testset['Score'].values
    helpfulness_truths = testset['NormalizedHelpfulness'].values

    Calc_F1(score_preds, score_truths, 'score')
    Calc_F1(helpfulness_preds, helpfulness_truths, 'help')

    resfp = 'res.' + args.model + '.csv'
    testset.to_csv(resfp, index=False)
    model.save_weights(model_file)
