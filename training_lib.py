#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/21 12:20
# @Author  : Kang
# @Site    : 
# @File    : training_lib.py
# @Software: PyCharm
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn import metrics
from data_process import get_task_data


def get_model_inputs(train=True, task='a', word2idx=None):
    assert word2idx != None
    tokenized_corpus, labels = get_task_data(train=train, task=task, word2idx=word2idx)
    max_len = np.max(np.array([len(sent) for sent in tokenized_corpus]))

    # we index our sentences
    vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in tokenized_corpus]
    # we create a tensor of a fixed size filled with zeroes for padding
    sent_tensor = Variable(torch.zeros((len(vectorized_sents), max_len))).long()
    sent_lengths = [len(sent) for sent in vectorized_sents]
    # we fill it with our vectorized sentences
    for idx, (sent, sentlen) in enumerate(zip(vectorized_sents, sent_lengths)):
        sent_tensor[idx, :sentlen] = torch.LongTensor(sent)
    if labels != None:
        if task == 'c':
            label_tensor = torch.LongTensor(labels)
        else:
            label_tensor = torch.FloatTensor(labels)
        return sent_tensor, label_tensor
    else:
        return sent_tensor, None


def accuracy(output, target):
    average = 'macro'
    length = len(target)
    if output.shape[-1] != 1:
        output = torch.argmax(output, dim=1).float()

    target = target.float()
    output = output.view(target.shape)
    predict = torch.round(output)
    correct = (predict == target).float()
    acc = correct.sum() / length

    y_true = np.array(target)
    y_pred = predict.detach().numpy()

    matrix = metrics.confusion_matrix(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average=average)
    return acc, f1_score, matrix


def train_valid_split(sent_tensor, label_tensor, valid_size=0.25, shuffle=True):
    # split dataset into training set and valid set
    length = len(label_tensor)
    if shuffle:
        p = np.random.permutation(length)
        sent_tensor, label_tensor = sent_tensor[p], label_tensor[p]
    split = int(length * (1 - valid_size))
    train_sent_tensor, train_label_tensor = sent_tensor[:split], label_tensor[:split]
    valid_sent_tensor, valid_label_tensor = sent_tensor[split:], label_tensor[split:]
    return train_sent_tensor, train_label_tensor, valid_sent_tensor, valid_label_tensor


def trainer(model, optimizer, loss_fn, sent_tensor, label_tensor, epoch_num=20, batch_size=32, valid_size=0.25, augment=None):
    # Trainer, get the model, loss function, optimizer, and train the model
    train_sent_tensor, train_label_tensor, valid_sent_tensor, valid_label_tensor = train_valid_split(sent_tensor,
                                                                                                     label_tensor,
                                                                                                     valid_size=valid_size)
    print(train_label_tensor.shape)
    if augment != None:
        tmp_tensor = train_sent_tensor[train_label_tensor == 0]
        train_sent_tensor = torch.cat([train_sent_tensor] + [tmp_tensor for i in range(augment)])
        train_label_tensor = torch.cat([train_label_tensor] + [torch.Tensor([0 for i in range(augment * len(tmp_tensor))])])
        assert len(train_sent_tensor) == len(train_label_tensor)

    batch_num = len(train_label_tensor) // batch_size
    if (len(train_label_tensor) % batch_size) != 0:
        batch_num += 1
    for epoch in range(1, epoch_num + 1):
        # shuffle the dataset
        p = np.random.permutation(len(train_label_tensor))
        train_sent_tensor, train_label_tensor = train_sent_tensor[p], train_label_tensor[p]
        epoch_loss = 0
        for i in range(batch_num):
            feature = train_sent_tensor[i * batch_size:(i + 1) * batch_size]
            target = train_label_tensor[i * batch_size:(i + 1) * batch_size]
            model.train()
            # we zero the gradients as they are not removed automatically
            optimizer.zero_grad()
            # queeze is needed as the predictions are initially size (batch size, 1) and we need to remove the dimension of size 1
            predictions = model(feature)
            if predictions.shape[-1] == 1:
                predictions = predictions.view(target.shape)
            loss = loss_fn(predictions, target)
            # calculate the gradient of each parameter
            loss.backward()
            # update the parameters using the gradients and optimizer algorithm
            optimizer.step()
            epoch_loss += loss.item()
        predict = model(train_sent_tensor)
        predict_val = model(valid_sent_tensor)
        train_acc, train_f1score, train_matrix = accuracy(predict, train_label_tensor)
        valid_acc, valid_f1score, valid_matrix = accuracy(predict_val, valid_label_tensor)
        print('Epoch: %3d | Train accuracy: %.2f%% | Valid acc: %.2f%%' % (epoch, train_acc * 100, valid_acc * 100))
        if predict.shape[-1] == 1:
            print('Epoch: %3d | Train f1_score: %.2f | Valid f1_score: %.2f' % (epoch, train_f1score, valid_f1score))
        else:
            print('Train f1score: ')
            print(train_f1score)
            print('Valid f1score: ')
            print(valid_f1score)
            print('Train confusion matrix: ')
        print(train_matrix)
        print('Valid confusion matrix: ')
        print(valid_matrix)
    return model


def generate_testing_result(model, word2idx, task='a'):
    test_sent_tensor, _ = get_model_inputs(train=False, task=task, word2idx=word2idx)

    model.eval()
    output = model(test_sent_tensor)
    if output.shape[-1] != 1:
        output = torch.argmax(output, dim=1).float()

    output = output.view(output.shape[0])
    pred_out = list(torch.round(output).detach().numpy())
    if task == 'a':
        test_path = 'OffensEval_task_data/Test A Release/testset-taska.tsv'
        predict = list(map(lambda x: 'OFF' if x == 1 else 'NOT', pred_out))
    elif task == 'b':
        test_path = 'OffensEval_task_data/Test B Release/testset-taskb.tsv'
        predict = list(map(lambda x: 'TIN' if x == 1 else 'UNT', pred_out))
    else:
        test_path = 'OffensEval_task_data/Test C Release/test_set_taskc.tsv'
        predict = list(map(lambda x: 'IND' if x == 0 else 'GRP' if x == 1 else 'OTH', pred_out))
    test_data = pd.read_csv(test_path, sep='\t', index_col=False)
    assert len(test_data) == len(pred_out)
    test_data['predict'] = predict
    csv_data = test_data[['id', 'predict']]
    csv_data.to_csv('TestResult/result_task' + task + '.csv', header=False, index=False)
    print('Test result for task ' + task + ' generated!')


