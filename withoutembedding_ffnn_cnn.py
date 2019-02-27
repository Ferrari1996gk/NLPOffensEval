#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 21:07
# @Author  : Kang
# @Site    : 
# @File    : simple_ffnn_cnn.py
# @Software: PyCharm
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# imprt self defined libray
from data_process import DataHandle, get_task_data
from training_lib import get_model_inputs, trainer, generate_testing_result

print('Libraries imported!')

# we fix the seeds to get consistent results
SEED = 234
torch.manual_seed(SEED)
np.random.seed(SEED)


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


def get_ffnn_model(Vocab_size, lr=0.01, task='a', EMBEDDING_DIM=50, OUTPUT_DIM=1):
    # the input dimension is the vocabulary size
    # we define our embedding dimension (dimensionality of the output of the first layer)
    # Hidden_dim: dimensionality of the output of the second hidden layer
    # OUTPUT_dim: the outut dimension is the number of classes, 1 for binary classification
    model = FFNN(vocab_size=Vocab_size, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # we use the Binary cross-entropy loss with sigmoid (applied to logits)
    # Recall we did not apply any activation to our output layer, we need to make our outputs look like probality.
    if task == 'c':
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


def get_cnn_model(Vocab_size, lr=0.01, task='a', EMBEDDING_DIM=50, OUTPUT_DIM=1, DROPOUT=0.):
    # the input dimension is the vocabulary size
    # we define our embedding dimension (dimensionality of the output of the first layer)
    # Hidden_dim: dimensionality of the output of the second hidden layer
    # OUTPUT_dim: the outut dimension is the number of classes, 1 for binary classification
    # we define the number of filters
    # we define the window size
    # we apply the dropout with the probability 0.2
    model = CNN(vocab_size=Vocab_size, embedding_dim=EMBEDDING_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # we use the Binary cross-entropy loss with sigmoid (applied to logits)
    # Recall we did not apply any activation to our output layer, we need to make our outputs look like probality.
    if task == 'c':
        loss_fn = nn.NLLLoss()
    else:
        loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


if __name__ == '__main__':
    import json
    with open('word2idx.json', 'r') as fp:
        word2idx = json.load(fp)
    # define your parameter
    task = 'a'
    embedding_dim = 50
    lr = 0.001
    output_dim = 3 if task == 'c' else 1
    train_sent_tensor, train_label_tensor = get_model_inputs(train=True, task=task, word2idx=word2idx)
    print(train_sent_tensor[:10])
    print('size of training set: ', train_sent_tensor.shape)

    print('For task ' + task)
    # print("Training on FFNN network!")
    # model, optimizer, loss_fn = get_ffnn_model(Vocab_size=len(word2idx), EMBEDDING_DIM=embedding_dim, lr=lr, OUTPUT_DIM=output_dim, task=task)
    print("training on CNN network!")
    model, optimizer, loss_fn = get_cnn_model(Vocab_size=len(word2idx), EMBEDDING_DIM=embedding_dim, lr=lr, OUTPUT_DIM=output_dim, task=task)
    trained_model = trainer(model, optimizer, loss_fn, train_sent_tensor, train_label_tensor,
            epoch_num=10, batch_size=32)
    generate_testing_result(model=trained_model, word2idx=word2idx, task=task)
