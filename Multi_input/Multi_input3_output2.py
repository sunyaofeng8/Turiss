import numpy as np

import tensorflow as tf
from tensorflow import keras

import pickle

class MultiIn3Out2:
    def __init__(self, trainset, testset, save=True, load=False, name1=None, name2=None, name3=None):
        self.trainset = trainset
        self.testset = testset
        self.save = save
        self.load = load
        self.name1 = name1
        self.name2 = name2
        self.name3 = name3

    # main function, return: model
    def getModel(self):
        y1 = self.ScoreToTensor(self.trainset['Score'].values)
        Y1 = self.ScoreToTensor(self.testset['Score'].values)
        y2 = self.ScoreToTensor(self.trainset['NormalizedHelpfulness'].values)
        Y2 = self.ScoreToTensor(self.testset['NormalizedHelpfulness'].values)
        train_Y = [y1, y2]
        test_Y = [Y1, Y2]

        if self.load:
            dic1 = self.Pkl2dic(self.name1)
            dic2 = self.Pkl2dic(self.name2)
            dic3 = self.Pkl2dic(self.name3)
        else:
            dic1 = self.Build_Dict('Product_ID')
            dic2 = self.Build_Dict('User_ID')
            dic3 = self.Build_Dict('Time_ID')

        if self.save:
            self.Save2Pkl(dic1, self.name1)
            self.Save2Pkl(dic2, self.name2)
            self.Save2Pkl(dic3, self.name3)

        x1 = self.RawToTensor(self.trainset['Product_ID'].values, dic1)
        X1 = self.RawToTensor(self.testset['Product_ID'].values, dic1)

        x2 = self.RawToTensor(self.trainset['User_ID'].values, dic2)
        X2 = self.RawToTensor(self.testset['User_ID'].values, dic2)

        x3 = self.RawToTensor(self.trainset['Time_ID'].values, dic3)
        X3 = self.RawToTensor(self.testset['Time_ID'].values, dic3)

        train_X = [x1, x2, x3]
        test_X = [X1, X2, X3]
        model = self.BuildModel(len(dic1), len(dic2), len(dic3))
        history = model.fit(x=train_X, y=train_Y, epochs=10, validation_data=(test_X, test_Y), shuffle=
        'steps_per_epoch')

        return model


    # save dic data to 'name.pkl'
    def Save2Pkl(self, dic, name):
        f = open(name+'.pkl', 'wb')
        pickle.dump(dic, f, -1)
        f.close()

    # given name of the pickle file, return: dic data in the file
    def Pkl2dic(self, name):
        f = open(name+'.pkl', 'rb')
        dic = pickle.load(f)
        f.close()
        return dic


    # given feature string, return: dic
    def Build_Dict(self, feature):
        count = {}
        data = list(self.trainset[feature].values) + list(self.testset[feature].values)

        for item in data:
            if not item in count:
                count[item] = 1
            else:
                count[item] += 1

        dic = {}
        lens = 0
        for item in count:
            if count[item] <= 2:  # Nan Threshold
                dic[item] = 0
            else:
                lens = lens + 1
                dic[item] = lens

        return dic

    def ScoreToTensor(self, raw_Y):
        Y = np.array(raw_Y) - 1  # convert to [0, 4]
        Y = [[int(t == label) for t in range(5)] for label in Y]
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)

        return Y

    # return: model
    def BuildModel(self, size1, size2, size3):
        hidden_size = 128
        data_size = 1

        input1 = keras.Input(shape=(data_size,), dtype=tf.float32)
        e1 = keras.layers.Embedding(size1, hidden_size, embeddings_initializer=tf.random_normal_initializer)(
            input1)

        input2 = keras.Input(shape=(data_size,), dtype=tf.float32)
        e2 = keras.layers.Embedding(size2, hidden_size, embeddings_initializer=tf.random_normal_initializer)(
            input2)

        input3 = keras.Input(shape=(data_size,), dtype=tf.float32)
        e3 = keras.layers.Embedding(size3, hidden_size, embeddings_initializer=tf.random_normal_initializer)(
            input3)

        merger1 = keras.layers.concatenate([e1, e2, e3])
        f1 = keras.layers.Flatten()(merger1)
        d1 = keras.layers.Dense(32, activation='relu')(f1)
        o1 = keras.layers.Dense(5, activation='relu')(d1)
        output_score = keras.layers.Dense(5, activation='softmax', name='output_score')(o1)

        merger2 = keras.layers.concatenate([e1, e2, e3])
        f2 = keras.layers.Flatten()(merger2)
        d2 = keras.layers.Dense(32, activation='relu')(f2)
        o2 = keras.layers.Dense(5, activation='relu')(d2)
        output_help = keras.layers.Dense(5, activation='softmax', name='output_help')(o2)

        model = keras.Model(inputs=[input1, input2, input3], outputs=[output_score, output_help])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        return model

    def RawToTensor(self, raw, dic):
        for item in range(len(raw)):
            raw[item] = dic[raw[item]]
            # print(dic1[raw[item]])
        return tf.convert_to_tensor(raw, dtype=tf.float32)


"""
preds = model.predict(test_X)
preds = preds.argmax(1)

truths = testset['Score'].values - 1

res = [[0] * 5 for i in range(5)]
for pred, truth in zip(preds, truths):
    res[truth][pred] += 1

tot = truths.size // 5

for i in range(5):
    print([x / tot for x in res[i]])

"""
