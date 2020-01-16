import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Comment(object):
    def __init__(self, seq, score, help):
        super(Comment, self).__init__()
        self.seq = seq
        self.score = score
        self.help = help


def getKey(x):
    return x.score * 100 + x.help

max_iter = 100
init_comments = 100

GT_data = pd.read_csv("../data/res.csv")
PR_data = pd.read_csv("../data/res.csv")

size_of_data = len(list(GT_data['Score'])) # size_of_data comments in the pool

# suppose we have list Seq, Score, and Help
pos_error = []
score_error = []

for iter in range(max_iter):
    GT_list = []
    PR_list = []
    for i in range(init_comments):  # randomly select init_comments comments
        index = np.random.randint(size_of_data)
        GT_list.append(Comment(GT_data['CleanedText'][index], GT_data['Score'][index], GT_data['NormalizedHelpfulness'][index]))

    GT_list.sort(key=getKey, reverse=True)
    PR_list = GT_list[:]

    index = np.random.randint(size_of_data)
    GT_add = Comment(GT_data['CleanedText'][index], GT_data['Score'][index], GT_data['NormalizedHelpfulness'][index])
    GT_score = GT_data['Score'][index]
    GT_list.append(GT_add)
    GT_list.sort(key=getKey, reverse=True)
    GT_position = GT_list.index(GT_add)
    GT_list.remove(GT_add)

    PR_add = Comment(PR_data['CleanedText'][index], PR_data['score_preds'][index], GT_data['NormalizedHelpfulness'][index])
    PR_score = PR_data['score_preds'][index]
    PR_list.append(PR_add)
    PR_list.sort(key=getKey, reverse=True)
    PR_position = PR_list.index(PR_add)
    PR_list.remove(PR_add)

    pos_error.append(GT_position-PR_position)
    score_error.append(GT_score-PR_score)

# print(score_error)

plt.figure()
sns.kdeplot(score_error, pos_error, cmap="Blues", shade=True, shade_lowest=False)
plt.xlabel('Score error')
plt.ylabel('System error (position)')
plt.title('System error induced by model error', color='royalblue')
plt.show()
