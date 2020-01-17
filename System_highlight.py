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

init_comments = 30

GT_data = pd.read_csv("./data/res.csv")
PR_data = pd.read_csv("./data/res.csv")

size_of_data = len(list(GT_data['Score'])) # size_of_data comments in the pool

# suppose we have list Seq, Score, and Help
pos_error = []
help_error = []

GT_list = []

for i in range(init_comments):  # randomly select init_comments comments
    index = np.random.randint(size_of_data)
    GT_list.append(Comment(GT_data['CleanedText'][index], GT_data['Score'][index],
                           GT_data['NormalizedHelpfulness'][index]))

BL_list = GT_list[:]
PR_list = GT_list[:]

index = np.random.randint(size_of_data)
GT_list.append(Comment(GT_data['CleanedText'][index], GT_data['Score'][index], GT_data['NormalizedHelpfulness'][index]))
BL_list.append(Comment(GT_data['CleanedText'][index], GT_data['Score'][index], 1))
PR_list.append(Comment(PR_data['CleanedText'][index], GT_data['Score'][index], PR_data['helpfulness_preds'][index]))
#print(PR_list[10].seq)
GT_list.sort(key=getKey, reverse=True)
BL_list.sort(key=getKey, reverse=True)
PR_list.sort(key=getKey, reverse=True)

TEXT = GT_data['CleanedText'][index]

#print(PR_list[10].seq)
#GT_csv = pd.DataFrame(columns=['comment', 'score', 'helpfulness'])

#BL_csv = pd.DataFrame(columns=['comment', 'score', 'helpfulness'])

#PR_csv = pd.DataFrame(columns=['comment', 'score', 'helpfulness'])
GT_csv = pd.DataFrame()
BL_csv = pd.DataFrame()
PR_csv = pd.DataFrame()
for i in range(1+init_comments):
    #print({"comment": GT_list[i].seq, "score": GT_list[i].score, "helpfullness": GT_list[i].help})
    GT_csv=GT_csv.append({"comment": GT_list[i].seq, "score": GT_list[i].score, "helpfullness": GT_list[i].help}, ignore_index=True)
    BL_csv=BL_csv.append({'comment': BL_list[i].seq, 'score': BL_list[i].score, 'helpfullness': BL_list[i].help}, ignore_index=True)
    PR_csv=PR_csv.append({'comment': PR_list[i].seq, 'score': PR_list[i].score, 'helpfullness': PR_list[i].help}, ignore_index=True)
#print(len(GT_csv))
print(TEXT)
GT_csv.to_csv('GT.csv')
BL_csv.to_csv('BL.csv')
PR_csv.to_csv('PR.csv')
