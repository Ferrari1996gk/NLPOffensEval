from expletives import badwords
from expletives import okayish

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import euclidean, cosine
from tqdm import tqdm 
import codecs
from sklearn.metrics.pairwise import cosine_distances
from data_process import DataHandle, get_task_data

datahandle=DataHandle()
# corpus without stemmer
tokenized_corpus=datahandle.tokenize()
bad_sen_ratio=0
accuracy=0
tp=0  # true positive/offensive
tn=0  # true negative
p=0  # all positive 
_, labels=get_task_data()
really_badwords = set(badwords).difference(okayish)
# badwords = really_badwords
print(len(labels), len(tokenized_corpus))
for i in range (len(tokenized_corpus)):

    if labels[i]==1:
        p+=1

    print(i ,'/', len(tokenized_corpus))

    for (idx, atom) in enumerate(tokenized_corpus[i]):
        if atom in badwords:
            if labels[i] == 1:
                accuracy += 1
                tp+=1
            bad_sen_ratio +=1
            break
        if idx >= len(tokenized_corpus[i])-1 and labels[i] == 0:
                tn += 1
                accuracy += 1
                break
n=len(tokenized_corpus)-p   # all negative 

print('bad_sen_ratio: ', bad_sen_ratio)
print('ture positive: {}/{}, rate: {}| true negative: {}/{}, rate: {}'.format(tp, p, tp/p, tn, n, tn/n))
print('accuracy: ', (accuracy/len(tokenized_corpus)))  

# print(sorted(badwords)[:10])
# if 'fuck' in badwords:
#     print('have')


# print(len(badwords))
# print(len(okayish))
# really_badwords = set(badwords).difference(okayish)
# print(len(really_badwords))