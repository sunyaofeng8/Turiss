import numpy as np
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


GT_data = pd.read_csv("../data/res_SingleLSTM.csv")
PR1_data = pd.read_csv("../data/res_SingleLSTM.csv")
PR2_data = pd.read_csv("../data/res_SingleBERT.csv")
PR3_data = pd.read_csv("../data/res_BERTMulModMulTask.csv")

size_of_data = len(list(GT_data['Score']))  # size_of_data comments in the pool

epsilon1 = 0  # epsilon = average distance from prediction to GT(of helpfulness, normalized to 0~1)
epsilon2 = 0
epsilon3 = 0
for iter in range(size_of_data):
    # '/4' to normalize helpfulness from 1~5 to 0~1
    epsilon1 += np.square(abs(GT_data['NormalizedHelpfulness'][iter] - PR1_data['helpfulness_preds'][iter]) / 4)
    epsilon2 += np.square(abs(GT_data['NormalizedHelpfulness'][iter] - PR2_data['helpfulness_preds'][iter]) / 4)
    epsilon3 += np.square(abs(GT_data['NormalizedHelpfulness'][iter] - PR3_data['helpfulness_preds'][iter]) / 4)
epsilon1 = np.sqrt(epsilon1 / size_of_data)
epsilon2 = np.sqrt(epsilon2 / size_of_data)
epsilon3 = np.sqrt(epsilon3 / size_of_data)

init_comments = 2500
model_equal1 = (int)(10 // np.square(epsilon1))
model_equal2 = (int)(10 // np.square(epsilon2))
model_equal3 = (int)(10 // np.square(epsilon3))

# time_step = 4 * max(model_equal1, model_equal2, model_equal3)
time_step = 400


GT_list = []
for i in range(init_comments):  # randomly select init_comments comments
    index = np.random.randint(size_of_data)
    GT_list.append(Comment(GT_data['CleanedText'][index], GT_data['Score'][index],
                           (GT_data['NormalizedHelpfulness'][index]-1)/4.0 +
                           np.random.normal(loc=0.0, scale=np.square(0.1), size=None)))  # add noise to GT_help for better sort
GT_list.sort(key=getKey, reverse=True)


# randomly choose a comment, record its Score(will not change) and Helpfulness
index = np.random.randint(size_of_data)  # randomly choose one new comment and initialize with helpfulness_preds
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
ave_help1 = 0
ave_help2 = 0
ave_help3 = 0
num_help = 0
for i in range(size_of_data):
    if (GT_data['NormalizedHelpfulness'][i]-1)/4.0 == GT_Help:
        ave_help1 += (PR1_data['helpfulness_preds'][i] - 1) / 4.0
        ave_help2 += (PR2_data['helpfulness_preds'][i] - 1) / 4.0
        ave_help3 += (PR3_data['helpfulness_preds'][i] - 1) / 4.0
        num_help += 1
ave_help1 = ave_help1 / num_help
ave_help2 = ave_help2 / num_help
ave_help3 = ave_help3 / num_help

p = GT_Help   # every iteration, user will regard the comment as helpful with P(helpful)=p

time_span = []
pos_error1 = []
pos_error2 = []
pos_error3 = []

print("epsilon1 = %f" % epsilon1)
print("epsilon2 = %f" % epsilon2)
print("epsilon3 = %f" % epsilon3)


for iter in range(time_step):
    u = np.random.binomial(1, p, size=None)

    # first model
    if iter == 0:   # first time, help = ave_help
        u = ave_help1
    ave_help1 = ave_help1 + (u - ave_help1) / (iter + model_equal1)
    Chosen_comment = Comment(Text, Score, ave_help1)
    GT_list.append(Chosen_comment)
    GT_list.sort(key=getKey, reverse=True)
    new_position = GT_list.index(Chosen_comment)
    GT_list.remove(Chosen_comment)

    pos_error1.append(abs(GT_position-new_position))

    # second model
    if iter == 0:
        u = ave_help2
    ave_help2 = ave_help2 + (u - ave_help2) / (iter + model_equal2)
    Chosen_comment = Comment(Text, Score, ave_help2)
    GT_list.append(Chosen_comment)
    GT_list.sort(key=getKey, reverse=True)
    new_position = GT_list.index(Chosen_comment)
    GT_list.remove(Chosen_comment)

    pos_error2.append(abs(GT_position - new_position))

    # third model
    if iter == 0:
        u = ave_help1
    ave_help3 = ave_help3 + (u - ave_help3) / (iter + model_equal3)
    Chosen_comment = Comment(Text, Score, ave_help3)
    GT_list.append(Chosen_comment)
    GT_list.sort(key=getKey, reverse=True)
    new_position = GT_list.index(Chosen_comment)
    GT_list.remove(Chosen_comment)

    pos_error3.append(abs(GT_position - new_position))

    time_span.append(iter)


plt.figure()
sns.lineplot(time_span, pos_error1, color='royalblue', label='SingleLSTM')  # model1
sns.lineplot(time_span, pos_error2, color='orangered', label='SingleBERT')  # model2
sns.lineplot(time_span, pos_error3, color='limegreen', label='BERTMulModMulTask')  # model3
plt.xlabel('Time span')
plt.ylabel('System error (position)')
plt.title('Model Comparison', color='royalblue')
plt.show()
