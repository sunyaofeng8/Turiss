import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, r'/Users/racoon727/Desktop/大学/Google/ML_winter_camp/Turiss-master/')

from Multi_input import MultiInput

trainset = pd.read_csv("../data/local_train_set.csv")
testset = pd.read_csv("../data/local_test_set.csv")

multi_input = MultiInput(trainset, testset, save=True, load=False, name1='dic_product', name2='dic_user', \
                   name3='dic_time', name4='dic_help_nume', name5='dic_help_deno')
model = multi_input.getModel()

print(model.summary())

