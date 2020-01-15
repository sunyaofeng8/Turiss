import numpy as np

import tensorflow as tf
from tensorflow import keras

import pickle

class MultiInput:
    def __init__(self, trainset, testset, save=True, load=False, name1=None, name2=None, name3=None, name4=None, name5=None):
        self.trainset = trainset
        self.testset = testset
        self.save = save
        self.load = load
        self.name1 = name1
        self.name2 = name2
        self.name3 = name3
        self.name4 = name4
        self.name5 = name5

    # main function, return: model
    def getModel(self):
        train_Y = self.ScoreToTensor(self.trainset['NormalizedHelpfulness'].values)
        test_Y = self.ScoreToTensor(self.testset['NormalizedHelpfulness'].values)

        if self.load:
            dic1 = self.Pkl2dic(self.name1)
            dic2 = self.Pkl2dic(self.name2)
            dic3 = self.Pkl2dic(self.name3)
            dic4 = self.Pkl2dic(self.name4)
            dic5 = self.Pkl2dic(self.name5)
        else:
            dic1 = self.Build_Dict('Product_ID')
            dic2 = self.Build_Dict('User_ID')
            dic3 = self.Build_Dict('Time_ID')
            dic4 = self.Build_Dict('HelpfulnessNumerator')
            dic5 = self.Build_Dict('HelpfulnessDenominator')

        if self.save:
            self.Save2Pkl(dic1, self.name1)
            self.Save2Pkl(dic2, self.name2)
            self.Save2Pkl(dic3, self.name3)
            self.Save2Pkl(dic4, self.name4)
            self.Save2Pkl(dic5, self.name5)

        x1 = self.RawToTensor(self.trainset['Product_ID'].values, dic1)
        X1 = self.RawToTensor(self.testset['Product_ID'].values, dic1)

        x2 = self.RawToTensor(self.trainset['User_ID'].values, dic2)
        X2 = self.RawToTensor(self.testset['User_ID'].values, dic2)

        x3 = self.RawToTensor(self.trainset['Time_ID'].values, dic3)
        X3 = self.RawToTensor(self.testset['Time_ID'].values, dic3)

        x4 = self.RawToTensor(self.trainset['HelpfulnessNumerator'].values, dic4)
        X4 = self.RawToTensor(self.testset['HelpfulnessNumerator'].values, dic4)

        x5 = self.RawToTensor(self.trainset['HelpfulnessDenominator'].values, dic5)
        X5 = self.RawToTensor(self.testset['HelpfulnessDenominator'].values, dic5)

        train_X = [x1, x2, x3, x4, x5]
        test_X = [X1, X2, X3, X4, X5]
        model = self.BuildModel(len(dic1), len(dic2), len(dic3), len(dic4), len(dic5))
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
    def BuildModel(self, size1, size2, size3, size4, size5):
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

        input4 = keras.Input(shape=(data_size,), dtype=tf.float32)
        e4 = keras.layers.Embedding(size4, hidden_size, embeddings_initializer=tf.random_normal_initializer)(
            input4)

        input5 = keras.Input(shape=(data_size,), dtype=tf.float32)
        e5 = keras.layers.Embedding(size5, hidden_size, embeddings_initializer=tf.random_normal_initializer)(
            input5)

        merger = keras.layers.concatenate([e1, e2, e3, e4, e5])
        f = keras.layers.Flatten()(merger)
        d = keras.layers.Dense(32, activation='relu')(f)
        o = keras.layers.Dense(5, activation='relu')(d)
        main_output = keras.layers.Dense(5, activation='softmax')(o)
        model = keras.Model(inputs=[input1, input2, input3, input4, input5], outputs=main_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
