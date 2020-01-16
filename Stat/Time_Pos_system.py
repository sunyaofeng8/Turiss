import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import binom

class Comment(object):
    def __init__(self, seq, score, help):
        super(Comment, self).__init__()
        self.seq = seq
        self.score = score
        self.help = help


def getKey(x):
    return x.score * 100 + x.help

GT_data = pd.read_csv("../data/res.csv")
PR_data = pd.read_csv("../data/res.csv")

size_of_data = len(list(GT_data['Score'])) # size_of_data comments in the pool

"""
epsilon = 0 # epsilon = average distance from prediction to GT(of helpfulness, normalized to 0~1)
for iter in range(size_of_data):
    # normalize helpfulness from 1~5 to 0~1
    # GT_data['NormalizedHelpfulness'][iter] = (GT_data['NormalizedHelpfulness'][iter]-1)/4.0
    # PR_data['helpfulness_preds'][iter] = (PR_data['helpfulness_preds'][iter]-1)/4.0
    epsilon += abs(GT_data['NormalizedHelpfulness'][iter] - PR_data['helpfulness_preds'][iter]) / 4
epsilon = epsilon / size_of_data

"""
init_comments = 2500
epsilon = 0.1
model_equal = (int)(1 // np.square(epsilon))
time_step = 4 * model_equal


GT_list = []
for i in range(init_comments):  # randomly select init_comments comments
    index = np.random.randint(size_of_data)
    GT_list.append(Comment(GT_data['CleanedText'][index], GT_data['Score'][index], \
                           (GT_data['NormalizedHelpfulness'][index]-1)/4.0+ \
                           np.random.normal(loc=0.0, scale=np.square(0.1), size=None))) # add noise to GT_help for better sort
GT_list.sort(key=getKey, reverse=True)


# randomly choose a comment, record its Score(will not change) and Helpfulness
index = np.random.randint(size_of_data) # randomly choose one new comment and initialize with helpfulness_preds
GT_Help = (GT_data['NormalizedHelpfulness'][index]-1)/4.0
GT_Help_noise = GT_Help + np.random.normal(loc=0.0, scale=np.square(0.1), size=None)
Chosen_comment = Comment(GT_data['CleanedText'][index], GT_data['Score'][index], GT_Help_noise)
GT_list.append(Chosen_comment)
GT_list.sort(key=getKey, reverse=True)
GT_position = GT_list.index(Chosen_comment)
Text = GT_data['CleanedText'][index]
Score = GT_data['Score'][index]
GT_list.remove(Chosen_comment)


# based on GT_Help, get the average PR_Help(ave_help)
ave_help = 0
num_help = 0
for i in range(size_of_data):
    if (GT_data['NormalizedHelpfulness'][i]-1)/4.0 == GT_Help:
        ave_help += (PR_data['helpfulness_preds'][i]-1)/4.0
        num_help += 1
ave_help = ave_help / num_help

# the first time of baseline
Chosen_comment = Comment(Text, Score, 0)
GT_list.append(Chosen_comment)
GT_list.sort(key=getKey, reverse=True)
first_position = GT_list.index(Chosen_comment)
GT_list.remove(Chosen_comment)

time_span2 = [0]
pos_error2 = [abs(GT_position-first_position)]
model_equal2 = model_equal
ave_help2 = 0



# the first time of model
Chosen_comment = Comment(Text, Score, ave_help)
GT_list.append(Chosen_comment)
GT_list.sort(key=getKey, reverse=True)
first_position = GT_list.index(Chosen_comment)
GT_list.remove(Chosen_comment)

new_position = first_position

p = GT_Help   # every iteration, user will regard the comment as helpful with P(helpful)=p


time_span1 = [0]
pos_error1 = [abs(GT_position-first_position)]
model_equal1 = model_equal
ave_help1 = ave_help


for iter in range(1, time_step):
    u = np.random.binomial(1, p, size=None)

    # sys1
    ave_help1 = ave_help1 + (u - ave_help1) / (iter + model_equal1)
    Chosen_comment = Comment(Text, Score, ave_help1)
    GT_list.append(Chosen_comment)
    GT_list.sort(key=getKey, reverse=True)
    new_position = GT_list.index(Chosen_comment)
    GT_list.remove(Chosen_comment)

    pos_error1.append(abs(GT_position-new_position))
    time_span1.append(iter)

    # sys2
    ave_help2 = ave_help2 + (u - ave_help2) / (iter + model_equal2)
    Chosen_comment = Comment(Text, Score, ave_help2)
    GT_list.append(Chosen_comment)
    GT_list.sort(key=getKey, reverse=True)
    new_position = GT_list.index(Chosen_comment)
    GT_list.remove(Chosen_comment)

    pos_error2.append(abs(GT_position - new_position))
    time_span2.append(iter)



plt.figure()
sns.lineplot(time_span1, pos_error1, color = 'royalblue', label = 'Our System') # model_equal = 1/epsilon^2
sns.lineplot(time_span2, pos_error2, color = 'orangered', label = 'Baseline System') # model_equal = 1/epsilon^2 * 10
# sns.lineplot(time_span3, pos_error3, color = 'limegreen', label = 'n=0.5/(ε^2)') # model_equal = 1/epsilon^2 // 10
plt.xlabel('Time span')
plt.ylabel('System error (position)')
plt.title('System rectification rate comparison', color='royalblue')
plt.show()
