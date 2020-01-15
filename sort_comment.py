import numpy as np
import tensorflow as tf
import matplotlib as mpl

class Comment(object):
    def __init__(self, seq, score, help):
        super(Comment, self).__init__()
        self.seq = seq
        self.score = score
        self.help = help


def getKey(x):
    return x.score * 100 + x.help


n = 30
a = ['good'] * n
b = np.random.randint(1, 6, (n,), int)
c = np.random.randint(1, 6, (n,), int)

comment_list = []
for i in range(n):
    comment_list.append(Comment(a[i], b[i], c[i]))

comment_list.sort(key=getKey, reverse=True)

for i in range(n):
    print(comment_list[i].seq, comment_list[i].score, comment_list[i].help)



