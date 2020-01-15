import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

train_df = pd.read_csv('amazon-fine-food-reviews/Reviews.csv')  # train set

data = train_df['HelpfulnessNumerator'] / train_df['HelpfulnessDenominator']

f = open("helpful_percentage.txt", 'w+')

print(data.value_counts(), file=f)


"""
# 取评价总人数[3,20)的数据，绝对值表格
data_three = train_df[(train_df['HelpfulnessDenominator']>3) & (train_df['HelpfulnessDenominator']<20)]
# data = data_three['HelpfulnessNumerator'] / data_three['HelpfulnessDenominator']
# f_three = open("helpful_percentage_three.txt", 'w+')
# print(data.value_counts(), file=f_three)

plt.figure()
sns.countplot(x='HelpfulnessDenominator', data=data_three, palette='RdBu')
plt.xlabel('HelpfulnessDenominator')
plt.show()

"""
"""
# 取评价总人数[3,20)的数据，loge表格
data_three = train_df[(train_df['HelpfulnessDenominator']>3) & (train_df['HelpfulnessDenominator']<20)]
data = data_three.HelpfulnessNumerator.value_counts()

data = np.log(data)

plt.figure()
sns.barplot(data.index, data.values)
plt.xlabel('HelpfulnessNumerator(log)')
plt.show()
"""


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

#train_df = train_df[(train_df['HelpfulnessDenominator']>0) & (train_df['HelpfulnessDenominator']<20000)]


train_df["Usefulness"] = (sigmoid(train_df['HelpfulnessDenominator']) * train_df["HelpfulnessNumerator"]/train_df["HelpfulnessDenominator"]).apply\
(lambda n: ">80%" if n >= 0.8 else ("<20%" if n < 0.2 else ("20-40%" if n >= 0.2 and n < 0.4 else ("40%-60%" if n >= 0.4\
and n < 0.6 else ("60-80%" if n >= 0.6 and n < 0.8 else "useless")) )))
# print(data)
# train_df["Usefulness"] = (train_df["HelpfulnessNumerator"]/train_df["HelpfulnessDenominator"])

'''
plt.figure()
# sns.barplot(data.index, data.values)
sns.countplot(x='Usefulness', data=train_df, palette='RdBu')
# sns.distplot(train_df.Usefulness)
plt.xlabel('Helpfulness')
plt.show()
'''

plt.figure()
sns.countplot(x='Score', data=train_df, palette='RdBu')
plt.xlabel('Score')
plt.show()

