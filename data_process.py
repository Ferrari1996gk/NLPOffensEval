#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 17:36
# @Author  : Kang
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()


def get_word2idx():
    print('Getting word2idx with train and test set------')
    train_path = 'OffensEval_task_data/start-kit/training-v1/offenseval-training-v1.tsv'
    testa_path = 'OffensEval_task_data/Test A Release/testset-taska.tsv'
    testb_path = 'OffensEval_task_data/Test B Release/testset-taskb.tsv'
    testc_path = 'OffensEval_task_data/Test C Release/test_set_taskc.tsv'
    train = pd.read_csv(train_path, sep='\t', index_col=False)
    testa = pd.read_csv(testa_path, sep='\t', index_col=False)
    testb = pd.read_csv(testb_path, sep='\t', index_col=False)
    testc = pd.read_csv(testc_path, sep='\t', index_col=False)
    raw_data = pd.concat([train[['id', 'tweet']], testa, testb, testc])
    tweet = raw_data.tweet
    corpus = list(tweet)
    tokenized_corpus = []
    for sentence in corpus:
        tmp_tokens = tokenizer.tokenize(sentence)
        lower_tokens = list(map(str.lower, tmp_tokens))
        tokenized_sentence = list(map(stemmer.stem, lower_tokens))
        tokenized_corpus.append(tokenized_sentence)
    vocabulary = []
    for sentence in tokenized_corpus:
        vocabulary += [token for token in sentence if token not in vocabulary]
    word2idx = {w: idx + 1 for (idx, w) in enumerate(vocabulary)}
    word2idx['<pad>'] = 0
    return word2idx, tokenized_corpus


class DataHandle:
    def __init__(self, path='OffensEval_task_data/start-kit/training-v1/offenseval-training-v1.tsv', word2idx=None):

        self.data_path = path
        self.raw_data = self.read_data()
        self.corpus = self.get_corpus()
        self.tokenized_corpus = self.tokenize()
        self.vocabulary = self.get_vocabulary()
        if word2idx != None:
            self.word2idx = word2idx
        else:
            self.word2idx = get_word2idx()

    def read_data(self):
        data = pd.read_csv(self.data_path, sep='\t', index_col=False)
        return data

    def get_corpus(self):
        print('------------Begin to get corpus-----------')
        tweet = self.raw_data.tweet
        corpus = list(tweet)
        return corpus

    def tokenize(self):
        print('------------Begin to tokenize corpus--------------')
        tokenized_corpus = []
        for sentence in self.corpus:
            tmp_tokens = tokenizer.tokenize(sentence)
            lower_tokens = list(map(str.lower, tmp_tokens))
            tokenized_sentence = list(map(stemmer.stem, lower_tokens))
            tokenized_corpus.append(tokenized_sentence)
        return tokenized_corpus

    def get_vocabulary(self):
        print('------------Begin to get vocabulary--------------')
        vocabulary = []
        for sentence in self.tokenized_corpus:
            vocabulary += [token for token in sentence if token not in vocabulary]
        return vocabulary


def get_task_data(train=True, task='a', word2idx=None):
    """
    To get the data for train/test and task a/b/c
    For training in task c, labels are one-hot encoded.
    :param train: if True, training data, if False, testing data
    :param task: 'a' / 'b' / 'c'
    :param word2idx: The total vocabulary.
    :return: traing: (tokenized corpus, train labels); test: (tokenized corpus, None)
    """
    print('---------------Prepare data for task '+task+'---------------')
    if train:
        print('---------You are requiring train data!---------')
        obj = DataHandle(word2idx=word2idx)
        all_text = obj.tokenized_corpus

        col = 'subtask_' + task
        if task == 'a':
            initial_labels = obj.raw_data[col].dropna().apply(lambda x: 1 if x == 'OFF' else 0)
        elif task == 'b':
            initial_labels = obj.raw_data[col].dropna().apply(lambda x: 1 if x == 'TIN' else 0)
        else:
            initial_labels = obj.raw_data[col].dropna().apply(lambda x: 0 if x == 'IND' else 1 if x == 'GRP' else 2)
        text = list(np.array(all_text)[list(initial_labels.index)])
        train_labels = list(initial_labels)
        return text, train_labels
    else:
        print('---------You are requiring test data!---------')
        if task == 'a':
            test_path = 'OffensEval_task_data/Test A Release/testset-taska.tsv'
        elif task == 'b':
            test_path = 'OffensEval_task_data/Test B Release/testset-taskb.tsv'
        else:
            test_path = 'OffensEval_task_data/Test C Release/test_set_taskc.tsv'
        obj = DataHandle(path=test_path, word2idx=word2idx)
        all_text = obj.tokenized_corpus
        return all_text, None

def onehot_encode(label):
    new_label = [[1, 0, 0] if x == 0 else [0, 1, 0] if x == 1 else [0, 0, 1] for x in label]
    return new_label

if __name__ == '__main__':
    import json
    word2idx, _ = get_word2idx()
    print(len(word2idx))
    with open('word2idx.json', 'w') as f:
        json.dump(word2idx, f)
        f.close()

    # ex = DataHandle(word2idx=word2idx)
    # print(len(ex.word2idx))
    data, label = get_task_data(word2idx=word2idx, task='c', train=False)
    print(data[20:25])
    print(label)
    # print(onehot_encode(label))
    # print(ex.vocabulary)
    # print(ex.word2idx)
