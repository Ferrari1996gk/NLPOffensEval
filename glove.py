
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import euclidean, cosine
from tqdm import tqdm 
import codecs
from sklearn.metrics.pairwise import cosine_distances
from data_process import DataHandle



def get_word2idx():
    w2i = [] # word2index
    i2w = [] # index2word
    wvecs = [] # word vectors

    # this is a large file, it will take a while to load in the memory!
    with codecs.open('./glove_data/glove.6B.50d.txt', 'r','utf-8') as f: 
      index = 0
      for line in tqdm(f.readlines()):
        # Ignore the first line - first line typically contains vocab, dimensionality
        if len(line.strip().split()) > 3:
          
          (word, vec) = (line.strip().split()[0], 
                        list(map(float,line.strip().split()[1:]))) 
          wvecs.append(vec)
          w2i.append((word, index))
          i2w.append((index, word))
          index += 1

    w2i = dict(w2i)
    i2w = dict(i2w)
    wvecs = np.array(wvecs)

    print(wvecs.shape)
    datahandle=DataHandle()
    tokenized_corpus=datahandle.tokenize()

    # adding words in our corpus to glove word2index: choose the index of unseen word to be 1
    for i in range (len(tokenized_corpus)):
      print(i ,'/', len(tokenized_corpus))
      for (_, atom) in enumerate(tokenized_corpus[i]):
        if atom not in w2i:
          w2i[atom]=1
          print(atom+' not in w2i!')

    return w2i, wvecs


import json
word2idx, embed_vectors = get_word2idx()
print(len(word2idx))
with open('word2idx_new.json', 'w') as f:
    json.dump(word2idx, f)
    f.close()

# np.save('embed_vectors.npy', embed_vectors)

