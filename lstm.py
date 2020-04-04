import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
from random import shuffle
from pre_trained_embedding import word_to_idx, vectors
import math
from collections import OrderedDict
import numpy as np 
import math
from Layer_function1 import *
from nonlinear import *
from img_TFRecords import *

from weight_loading import load_from_tflite, load_from_pb
from Weight_mapping import *
from gl import *

# import tensorflow as tf
import os
import h5py
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class Sentiment(torch.nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    
    def __init__(self):
        super(Sentiment, self).__init__()
        self.embed = nn.Embedding(400001, 50)
        # self.embed.weight = Variable(vectors)
        # self.embed.weight.requires_grad = False
        self.rnn = nn.LSTM(50, 50, batch_first=True)
        self.fc = nn.Linear(50, 2)
        self.smd = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embed(x)
        # x = x * 20 / torch.sum(mask, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = torch.mean(x, 1)
        x = self.smd(x)
        return x

def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            embedded = torch.nn.Embedding(inputs)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return 100 * correct / total

def prediction(lines_p, input_p, net, device):
    with torch.no_grad():
        input_p = input_p.to(device)
        outputs = net(input_p)
        _, predicted = torch.max(outputs.data, 1)
        f = open('predictions_4.txt', 'a')
        for i in range(len(lines_p)):
            f.write(str(predicted.numpy()[i]) + '\n')
        f.close()         

def pad(tensor, length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #generate input vectors
    f_train = open('data/train.txt')
    lines_train = f_train.readlines()
    f_train.close()
    f_train = open('data/train.txt')
    words = f_train.read().split()
    f_train.close()

    vocab = set(words)
    vocab = list(vocab)
    vocab.remove('0')
    vocab.remove('1')
    
    
    f_val = open('data/dev.txt')
    lines_val = f_val.readlines()
    f_val.close()

    f_test = open('data/test.txt')
    lines_test = f_test.readlines()
    f_test.close()

    f_p = open('data/unlabelled.txt')
    lines_p = f_p.readlines()
    f_p.close()

    

    # shuffle(lines_test)
    # word2vec = {}
    # for line in pre_trained:
    #     words = line.split()
    #     word2vec[words[0]] = list(words[1:], dtype=float)
    
    # word_to_ix = {word: i for i, word in enumerate(vocab)}
    input_train = []
    label_train = torch.zeros(len(lines_train))
    len_max = 0
    len_min = len(lines_train[0].split()) - 1
    # offset_train = []
    for i in range(len(lines_train)):
        label_train[i] = int(lines_train[i].split()[0])
        input_line = lines_train[i].split()
        input_line_new = []
        for word in input_line[1:]:
            if word in word_to_idx.keys():
                input_line_new.append(word_to_idx[word])
        input_train.append(torch.tensor(input_line_new))
        if len_max < len(input_train[i]):
            len_max = len(input_train[i])
        if len_min > len(input_train[i]):
            len_min = len(input_train[i])
    label_train = label_train.long()
    # offset_train = torch.tensor(offset_train, dtype=torch.long)
    # input_train = torch.tensor(input_train, dtype=torch.long)
    
    # offset_val = []
    input_val = []
    label_val = torch.zeros(len(lines_val))
    # start = 0
    
    for i in range(len(lines_val)):
        label_val[i] = int(lines_val[i].split()[0])
        input_line = lines_val[i].split()
        # offset_val.append(start)
        input_line_new = []
        for word in input_line[1:]:
            if word in word_to_idx.keys():
                input_line_new.append(word_to_idx[word])
        input_val.append(torch.tensor(input_line_new))
        
        if len_max < len(input_val[i]):
            len_max = len(input_val[i])
        if len_min > len(input_train[i]):
            len_min = len(input_train[i])
    label_val = label_val.long()
    # offset_val = torch.tensor(offset_val, dtype=torch.long)
    # input_val = torch.tensor(input_val, dtype=torch.long)
    
    # offset_test = []
    input_test = []
    label_test = torch.zeros(len(lines_test))
    start = 0
    for i in range(len(lines_test)):
        label_test[i] = int(lines_test[i].split()[0])
        input_line = lines_test[i].split()
        # offset_test.append(start)
        input_line_new = []
        for word in input_line[1:]:
            if word in word_to_idx.keys():
                input_line_new.append(word_to_idx[word])
        input_test.append(torch.tensor(input_line_new))
        
        if len_max < len(input_test[i]):
            len_max = len(input_test[i])
        if len_min > len(input_train[i]):
            len_min = len(input_train[i])
    label_test = label_test.long()
    # offset_test = torch.tensor(offset_test, dtype=torch.long)
    # input_test = torch.tensor(input_test, dtype=torch.long)

    input_p = []
    # offset_p = []
    start = 0
    for i in range(len(lines_p)):
        input_line = lines_p[i].split()
        # offset_p.append(start)
        input_line_new = []
        for word in input_line[1:]:
            if word in word_to_idx.keys():
                input_line_new.append(word_to_idx[word])
        input_p.append(torch.tensor(input_line_new))
        if len_max < len(input_p[i]):
            len_max = len(input_p[i])
        if len_min > len(input_train[i]):
            len_min = len(input_train[i])
    print(len_max)
    print(len_min)


    # padded_train = torch.stack([pad(input_train[i], len_max) for i in range(len(input_train))])
    # padded_val = torch.stack([pad(input_val[i], len_max) for i in range(len(input_val))])
    padded_test = torch.stack([pad(input_test[i], len_max) for i in range(len(input_test))])
    # padded_p = torch.stack([pad(input_p[i], len_max) for i in range(len(input_p))])

    # trainset = torch.utils.data.TensorDataset(padded_train, label_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
    #                                       shuffle=True)

    # valset = torch.utils.data.TensorDataset(padded_val, label_val)
    # valloader = torch.utils.data.DataLoader(valset, batch_size = 100, shuffle=False)

    testset = torch.utils.data.TensorDataset(padded_test, label_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    RNN(testloader, lines_test)
    # net = Sentiment().to(device)
    # net.embed.weight.data.copy_(vectors)
    # # net.embed.weight.requires_grad = False
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0)

    # train(trainloader, valloader, net, criterion, optimizer, device)
    # # train(input_train, input_val, label_train, label_val, offset_train, offset_val, net, criterion, optimizer, device)

    # test_acc = test(testloader, net, device)
    # print('Accuracy of the network on the 10000 test images: %d %%' % (test_acc))
    # prediction(lines_p, padded_p, net, device)


def RNN(testloader, lines_test):
    act_bit = 8
    model = torch.load('lstm_model.pt')
    embed_weight = model.embed.weight.data.numpy()
    lstm_ih_weight = model.rnn.weight_ih_l0.data.numpy().T
    lstm_hh_weight = model.rnn.weight_hh_l0.data.numpy().T
    lstm_ih_bias = model.rnn.bias_ih_l0.data.numpy()
    lstm_hh_bias = model.rnn.bias_hh_l0.data.numpy()

    w_ii = lstm_ih_weight[:,:50]
    w_if = lstm_ih_weight[:,50:100]
    w_ig = lstm_ih_weight[:,100:150]
    w_io = lstm_ih_weight[:,150:]

    w_hi = lstm_hh_weight[:,:50]
    w_hf = lstm_hh_weight[:,50:100]
    w_hg = lstm_hh_weight[:,100:150]
    w_ho = lstm_hh_weight[:,150:]

    b_ii = lstm_ih_bias[:50]
    b_if = lstm_ih_bias[50:100]
    b_ig = lstm_ih_bias[100:150]
    b_io = lstm_ih_bias[150:]

    b_hi = lstm_hh_bias[:50]
    b_hf = lstm_hh_bias[50:100]
    b_hg = lstm_hh_bias[100:150]
    b_ho = lstm_hh_bias[150:]

    fc_weight = model.fc.weight.data.numpy().T
    fc_bias = model.fc.bias.data.numpy()

    # scale_w_ii = (max(abs(np.max(w_ii)), abs(np.min(w_ii))) / 127)
    # scale_w_if = (max(abs(np.max(w_if)), abs(np.min(w_if))) / 127)
    # scale_w_ig = (max(abs(np.max(w_ig)), abs(np.min(w_ig))) / 127)
    # scale_w_io = (max(abs(np.max(w_io)), abs(np.min(w_io))) / 127)

    # scale_w_hi = (max(abs(np.max(w_hi)), abs(np.min(w_hi))) / 127)
    # scale_w_hf = (max(abs(np.max(w_hf)), abs(np.min(w_hf))) / 127)
    # scale_w_hg = (max(abs(np.max(w_hg)), abs(np.min(w_hg))) / 127)
    # scale_w_ho = (max(abs(np.max(w_ho)), abs(np.min(w_ho))) / 127)

    scale_ih_weight = (max(abs(np.max(lstm_ih_weight)), abs(np.min(lstm_ih_weight))) / (2**(act_bit-1)-1))
    scale_hh_weight = (max(abs(np.max(lstm_hh_weight)), abs(np.min(lstm_hh_weight))) / (2**(act_bit-1)-1))
    scale_fc_weight = (max(abs(np.max(fc_weight)), abs(np.min(fc_weight))) / (2**(act_bit-1)-1))

    # w_ii = np.round(w_ii / scale_w_ii)
    # w_if = np.round(w_if / scale_w_if)
    # w_ig = np.round(w_ig / scale_w_ig)
    # w_io = np.round(w_io / scale_w_io)

    # w_hi = np.round(w_hi / scale_w_hi)
    # w_hf = np.round(w_hf / scale_w_hf)
    # w_hg = np.round(w_hg / scale_w_hg)
    # w_ho = np.round(w_ho / scale_w_ho)

    # w_ii = np.round(w_ii / scale_ih_weight)
    # w_if = np.round(w_if / scale_ih_weight)
    # w_ig = np.round(w_ig / scale_ih_weight)
    # w_io = np.round(w_io / scale_ih_weight)

    # w_hi = np.round(w_hi / scale_hh_weight)
    # w_hf = np.round(w_hf / scale_hh_weight)
    # w_hg = np.round(w_hg / scale_hh_weight)
    # w_ho = np.round(w_ho / scale_hh_weight)

    lstm_ih_weight = np.round(lstm_ih_weight / scale_ih_weight)
    lstm_hh_weight = np.round(lstm_hh_weight / scale_hh_weight)
    fc_weight = np.round(fc_weight / scale_fc_weight)

    # w_ii_mapped = fc_mapping(w_ii)
    # w_if_mapped = fc_mapping(w_if)
    # w_ig_mapped = fc_mapping(w_ig)
    # w_io_mapped = fc_mapping(w_io)

    # w_hi_mapped = fc_mapping(w_hi)
    # w_hf_mapped = fc_mapping(w_hf)
    # w_hg_mapped = fc_mapping(w_hg)
    # w_ho_mapped = fc_mapping(w_ho)

    weight_ih_mapped = fc_mapping(lstm_ih_weight)
    weight_hh_mapped = fc_mapping(lstm_hh_weight)
    fc_weight_mapped = fc_mapping(fc_weight)


    input_dim, hidden_dim = w_ii.shape
    total = 0
    correct = 0
    for data in testloader:
        line = lines_test[total][2:-1]
        total += 1
        inputs, labels = data
        embed = np.squeeze(torch.embedding(torch.tensor(embed_weight), inputs).numpy())[:,np.newaxis,:]
        inputs = inputs.numpy()
        
        labels = labels.numpy()
        _, time_step = inputs.shape
        hidden_prev = np.zeros((1, hidden_dim))
        c_prev = np.zeros((1, hidden_dim))
        output_rnn = np.zeros((time_step, hidden_dim))
        
        scale_input = max(abs(np.max(embed)), abs(np.min(embed))) / (2**(act_bit-1)-1)
        
        # b_ii_q = b_ii / scale_w_ii / scale2bit(scale_input)
        # b_if_q = b_if / scale_w_if / scale2bit(scale_input)
        # b_ig_q = b_ig / scale_w_ig / scale2bit(scale_input)
        # b_io_q = b_io / scale_w_io / scale2bit(scale_input)

        # b_hi_q = b_hi / scale_w_hi / scale2bit(scale_input)
        # b_hf_q = b_hf / scale_w_hf / scale2bit(scale_input)
        # b_hg_q = b_hg / scale_w_hg / scale2bit(scale_input)
        # b_ho_q = b_ho / scale_w_ho / scale2bit(scale_input)

        # b_ii_q = b_ii / scale_ih_weight / scale2bit(scale_input)
        # b_if_q = b_if / scale_ih_weight / scale2bit(scale_input)
        # b_ig_q = b_ig / scale_ih_weight / scale2bit(scale_input)
        # b_io_q = b_io / scale_ih_weight / scale2bit(scale_input)

        # b_hi_q = b_hi / scale_hh_weight / scale2bit(scale_input)
        # b_hf_q = b_hf / scale_hh_weight / scale2bit(scale_input)
        # b_hg_q = b_hg / scale_hh_weight / scale2bit(scale_input)
        # b_ho_q = b_ho / scale_hh_weight / scale2bit(scale_input)
        lstm_ih_bias_q = lstm_ih_bias / scale_ih_weight / (scale_input)
        lstm_hh_bias_q = lstm_hh_bias / scale_hh_weight / (scale_input)

        embed = np.round(embed/ (scale_input))
        for t in range(time_step):
            scale_hidden = max(abs(np.max(hidden_prev)), abs(np.min(hidden_prev))) / (2**(act_bit-1)-1)
            scale_c = max(abs(np.max(c_prev)), abs(np.min(c_prev))) / (2**(act_bit-1)-1)
            if scale_hidden == 0:
                scale_hidden = 0.001
            if scale_c == 0:
                scale_c = 0.001
            hidden_prev = np.round(hidden_prev/ (scale_hidden))
            # c_prev = np.round(c_prev/ scale2bit(scale_c / (2**(act_bit-1)-1)))
            # i = sigmoid(fc_int(embed[t], scale_input, w_ii, w_ii_mapped, scale_w_ii, b_ii_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_hi, w_hi_mapped, scale_w_hi, b_hi_q, 8, True))
            # f = sigmoid(fc_int(embed[t], scale_input, w_if, w_if_mapped, scale_w_if, b_if_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_hf, w_hf_mapped, scale_w_hf, b_hf_q, 8, True))
            # g = tanh(fc_int(embed[t], scale_input, w_ig, w_ig_mapped, scale_w_ig, b_ig_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_hg, w_hg_mapped, scale_w_hg, b_hg_q, 8, True))
            # o = sigmoid(fc_int(embed[t], scale_input, w_io, w_io_mapped, scale_w_io, b_io_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_ho, w_ho_mapped, scale_w_ho, b_ho_q, 8, True))
            tmp = fc_int(embed[t], scale_input, lstm_ih_weight, weight_ih_mapped, scale_ih_weight, lstm_ih_bias_q, 8, True) \
                  + fc_int(hidden_prev, scale_hidden, lstm_hh_weight, weight_hh_mapped, scale_hh_weight, lstm_hh_bias_q, 8, True)
            i = sigmoid(tmp[:,:50])
            f = sigmoid(tmp[:,50:100])
            g = tanh(tmp[:,100:150])
            o = sigmoid(tmp[:,150:])
            # i = sigmoid(fc_int(embed[t], scale_input, w_ii, w_ii_mapped, scale_ih_weight, b_ii_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_hi, w_hi_mapped, scale_hh_weight, b_hi_q, 8, True))
            # f = sigmoid(fc_int(embed[t], scale_input, w_if, w_if_mapped, scale_ih_weight, b_if_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_hf, w_hf_mapped, scale_hh_weight, b_hf_q, 8, True))
            # g = tanh(fc_int(embed[t], scale_input, w_ig, w_ig_mapped, scale_ih_weight, b_ig_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_hg, w_hg_mapped, scale_hh_weight, b_hg_q, 8, True))
            # o = sigmoid(fc_int(embed[t], scale_input, w_io, w_io_mapped, scale_ih_weight, b_io_q, 8, True) \
            #     + fc_int(hidden_prev, scale_hidden, w_ho, w_ho_mapped, scale_hh_weight, b_ho_q, 8, True))
            
            c = f * c_prev + i * g
            output_rnn[t] = o * tanh(c)
            hidden_prev = output_rnn[t][np.newaxis,:]                                                      
            c_prev = c
        scale_fc = max(abs(np.max(output_rnn)), abs(np.min(output_rnn))) / (2**(act_bit-1)-1)
        if scale_fc == 0:
            scale_fc = 0.001
        fc_bias_q = fc_bias / scale_fc_weight / (scale_fc)
        output_rnn = np.round(output_rnn/ (scale_fc))
        output_fc = fc_int(output_rnn, scale_fc, fc_weight, fc_weight_mapped, scale_fc_weight, fc_bias_q, 2, True)
        # output_fc = fc(output_rnn, fc_weight, fc_weight_mapped) + fc_bias
        outputs = np.mean(output_fc, axis=0)
        prediction = np.argmax(outputs)
        print(line)
        print('label='+str(int(data[-1][0]))+', prediction='+str(prediction)+'\n')
        if prediction == labels:
            correct += 1
    accuracy = correct / total
    print(accuracy)
main()