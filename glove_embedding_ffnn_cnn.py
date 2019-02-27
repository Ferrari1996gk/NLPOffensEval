#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 23:22
# @Author  : Kang
# @Site    : 
# @File    : embedding_ffnn_cnn.py
# @Software: PyCharm
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gensim.models import Word2Vec
# imprt self defined libray
from data_process import DataHandle, get_task_data, get_word2idx
from training_lib import get_model_inputs, trainer

print('Libraries imported!')

# we fix the seeds to get consistent results
SEED = 234
torch.manual_seed(SEED)
np.random.seed(SEED)


# def word2vec_embedding(tokenized_corpus, embed_size=50, min_count=1, window=5):
#     sentences = tokenized_corpus
#     model = Word2Vec(sentences,min_count=min_count, window=window, size=embed_size)
#     # model.build_vocab(sentences)  # prepare the model vocabulary
#     # train word vectors
#     model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
#     # add the first vector as pading
#     embed_vectors = np.vstack([np.zeros((1, embed_size)), model.wv.vectors])
#     print('embed_vectors size: ', embed_vectors.shape )
#     return embed_vectors


def word2vec_embedding():
    return np.load('embed_vectors.npy')



class FFNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(FFNN, self).__init__()
        hidden_dim = 50
        # embedding (lookup layer) layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # activation
        self.relu1 = nn.ReLU()
        # output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        if output_dim == 1:
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        # print(embedded.shape)
        # we average the embeddings of words in a sentence
        averaged = torch.mean(embedded, dim=1)
        # (batch size, max sent length, embedding dim) to (batch size, embedding dim)
        out = self.fc1(averaged)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.out_act(out)
        return out


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout):
        super(CNN, self).__init__()
        out_channels = 150
        window_size = 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # in_channels -- 1 text channel
        # out_channels -- the number of output channels
        # kernel_size is (window size x embedding dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(window_size, embedding_dim))
        # the dropout layer
        self.dropout = nn.Dropout(dropout)
        # the output layer
        self.fc = nn.Linear(out_channels, output_dim)
        if output_dim == 1:
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # (batch size, max sent length)
        embedded = self.embedding(x)
        # (batch size, max sent length, embedding dim)
        # images have 3 RGB channels
        # for the text we add 1 channel
        embedded = embedded.unsqueeze(1)
        # (batch size, 1, max sent length, embedding dim)
        feature_maps = self.conv(embedded)
        # Q. what is the shape of the convolution output ?
        feature_maps = feature_maps.squeeze(3)
        # Q. why do we reduce 1 dimention here ?
        feature_maps = F.relu(feature_maps)
        # the max pooling layer
        pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])
        pooled = pooled.squeeze(2)
        # Q. what is the shape of the pooling output ?
        dropped = self.dropout(pooled)
        preds = self.fc(dropped)
        # preds = torch.sigmoid(preds)
        preds = self.out_act(preds)
        return preds



def embed_ffnn_model(embedding, Vocab_size, lr=0.01, task='a', EMBEDDING_DIM=50, OUTPUT_DIM=1):
    # embedding: np.array
    # we define our embedding dimension (dimensionality of the output of the first layer)
    # Hidden_dim: dimensionality of the output of the second hidden layer
    # OUTPUT_dim: the outut dimension is the number of classes, 1 for binary classification
    assert embedding.shape[1] == EMBEDDING_DIM
    model = FFNN(vocab_size=Vocab_size, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM)
    model.embedding.weight.data.copy_(torch.from_numpy(embedding))
    model.embedding.weight.require_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # we use the Binary cross-entropy loss with sigmoid (applied to logits)
    # Recall we did not apply any activation to our output layer, we need to make our outputs look like probality.
    if task == 'c':
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


def embed_cnn_model(embedding, Vocab_size, lr=0.08, task='a', EMBEDDING_DIM=50, OUTPUT_DIM=1, DROPOUT=0.):
    # embedding: np.array
    # we define our embedding dimension (dimensionality of the output of the first layer)
    # Hidden_dim: dimensionality of the output of the second hidden layer
    # OUTPUT_dim: the outut dimension is the number of classes, 1 for binary classification
    # we define the number of filters
    # we define the window size
    assert embedding.shape[1] == EMBEDDING_DIM
    model = CNN(vocab_size=Vocab_size, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    model.embedding.weight.data.copy_(torch.from_numpy(embedding))
    model.embedding.weight.require_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # we use the Binary cross-entropy loss with sigmoid (applied to logits)
    # Recall we did not apply any activation to our output layer, we need to make our outputs look like probality.
    if task == 'c':
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


if __name__ == '__main__':
    # import json
    # with open('word2idx.json', 'r') as fp:
    #     word2idx = json.load(fp)
    # word2idx, tokenized_corpus = get_word2idx()

    import json
    with open('word2idx_new.json', 'r') as fp:
        word2idx = json.load(fp)
    _, tokenized_corpus = get_word2idx()


    # print(word2idx['<pad>'])
    # define your own parameter.
    task = 'a'
    embedding_dim = 50

    output_dim = 3 if task == 'c' else 1
    train_sent_tensor, train_label_tensor = get_model_inputs(train=True, task=task, word2idx=word2idx)
    print('size of training set: ', train_sent_tensor.shape)


    print('Begin to get word2vec ')
    # embedding = word2vec_embedding(tokenized_corpus, embed_size=50)
    embedding = word2vec_embedding()
    # size of vocabulary in Glove embedding
    length_embedding = 400000

    print("Training on FFNN network with embedding!")
    model, optimizer, loss_fn = embed_ffnn_model(embedding, Vocab_size=length_embedding, EMBEDDING_DIM=embedding_dim, lr=0.001, OUTPUT_DIM=output_dim, task=task)
    # print("training on CNN network with embedding!")
    # model, optimizer, loss_fn = embed_cnn_model(embedding, Vocab_size=len(word2idx), EMBEDDING_DIM=embedding_dim, lr=0.001, OUTPUT_DIM=output_dim, task=task)
    trainer(model, optimizer, loss_fn, train_sent_tensor, train_label_tensor,
            epoch_num=20, batch_size=32)
