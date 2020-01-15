import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
#src_path = 'data/train_set.csv'  # preprocessed  training set
src_path = 'data/test_set.csv'
#tgt_path = 'data/train_helpfulness.csv'
tgt_path = 'data/test_helpfulness.csv'
src_df = pd.read_csv(src_path)
tgt_f = open(tgt_path, 'w+')

print(src_df.head())
# Product_ID,User_ID,Time_ID,HelpfulnessNumerator,HelpfulnessDenominator,CleanedText,Score
numerator = src_df['HelpfulnessNumerator']
denominator = src_df['HelpfulnessDenominator']


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

#src_df = src_df[(src_df['HelpfulnessDenominator']>0) & (src_df['HelpfulnessDenominator']<20000)]


epsilon = 1e-5  # avoid zero denominators
src_df['NormalizedHelpfulness'] = 1 + (5 * sigmoid(src_df['HelpfulnessDenominator']) * src_df['HelpfulnessNumerator']/
                                      (epsilon+src_df['HelpfulnessDenominator'])).apply(int)
src_df.to_csv(tgt_f)



