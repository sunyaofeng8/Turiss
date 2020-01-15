import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, r'/Users/racoon727/Desktop/大学/Google/ML_winter_camp/Turiss-master/')

from Multi_input_3 import MultiInput3

trainset = pd.read_csv("../data/train_helpfulness.csv")
testset = pd.read_csv("../data/test_helpfulness.csv")

multi_input = MultiInput3 (trainset, testset, save=True, load=False, name1='newdic_product', name2='newdic_user', \
                   name3='newdic_time')
model = multi_input.getModel()

print(model.summary())

