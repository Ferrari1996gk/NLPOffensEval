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
from training_lib import get_model_inputs, trainer, generate_testing_result

print('Libraries imported!')

# we fix the seeds to get consistent results
SEED = 234
torch.manual_seed(SEED)
np.random.seed(SEED)


def word2vec_embedding(tokenized_corpus, embed_size=50, min_count=1, window=5):
    sentences = tokenized_corpus
    model = Word2Vec(sentences, min_count=min_count, window=window, size=embed_size)
    # model.build_vocab(sentences)  # prepare the model vocabulary
    # train word vectors
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    # add the first vector as pading
    embed_vectors = np.vstack([np.zeros((1, embed_size)), model.wv.vectors])
    vocabulary = ['<pad>'] + model.wv.index2word
    return embed_vectors, vocabulary


class FFNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(FFNN, self).__init__()
        hidden_dim = 50
        # embedding (lookup layer) layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 16)
        # activation
        self.relu = nn.ReLU()
        # output layer
        self.fc2 = nn.Linear(16, output_dim)
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
        out = self.relu(out)
        out = self.fc2(self.relu(self.fc3(out)))
        out = self.out_act(out)
        return out


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout):
        super(CNN, self).__init__()
        out_channels = 128
        window_size = 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # in_channels -- 1 text channel
        # out_channels -- the number of output channels
        # kernel_size is (window size x embedding dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(window_size, embedding_dim), stride=1)
        self.pooling = nn.MaxPool1d(105)
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
        # pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])
        pooled = self.pooling(feature_maps)
        pooled = pooled.squeeze(2)
        # print(feature_maps.shape[2])
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
        loss_fn = nn.NLLLoss(weight=torch.Tensor([1.0, 2.0, 8.0]))
    else:
        loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


def embed_cnn_model(embedding, Vocab_size, lr=0.01, task='a', EMBEDDING_DIM=50, OUTPUT_DIM=1, DROPOUT=0.):
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
        loss_fn = nn.NLLLoss(weight=torch.Tensor([1.0, 2.0, 8.0]))
    else:
        loss_fn = nn.BCELoss()
    return model, optimizer, loss_fn


if __name__ == '__main__':
    # To be consistent with word2vec, we donot get word2idx here.
    _, tokenized_corpus = get_word2idx()

    # define your own parameter.
    task = 'c'
    embedding_dim = 50
    lr = 0.001
    output_dim = 3 if task == 'c' else 1

    print('Begin to get word2vec ')
    embedding, vocabulary = word2vec_embedding(tokenized_corpus, embed_size=embedding_dim)
    # Get word2idx according to word2vec.
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}

    train_sent_tensor, train_label_tensor = get_model_inputs(train=True, task=task, word2idx=word2idx)
    print('size of training set: ', train_sent_tensor.shape)


    # print("Training on FFNN network with embedding!")
    # model, optimizer, loss_fn = embed_ffnn_model(embedding, Vocab_size=len(word2idx), EMBEDDING_DIM=embedding_dim, lr=lr, OUTPUT_DIM=output_dim, task=task)
    print("training on CNN network with embedding!")
    model, optimizer, loss_fn = embed_cnn_model(embedding, Vocab_size=len(word2idx), EMBEDDING_DIM=embedding_dim, lr=lr, OUTPUT_DIM=output_dim, task=task)

    print(model)
    trained_model = trainer(model, optimizer, loss_fn, train_sent_tensor, train_label_tensor,
            epoch_num=10, batch_size=32, augment=None)
    generate_testing_result(model=trained_model, word2idx=word2idx, task=task)
