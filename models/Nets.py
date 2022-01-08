#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        # return self.softmax(x)
        return self.relu(x)

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCIFAR1(nn.Module):
    def __init__(self, args):
        super(CNNCIFAR1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pooling1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pooling2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(4096, 512)
        self.drop3 = nn.Dropout2d(p=0.25)
        self.fc3 = nn.Linear(512, 10)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.batchNorm4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.batchNorm1( F.relu( self.conv1( x)))
        x = self.pooling1(self.batchNorm2( F.relu( self.conv2( x))))
        x = self.batchNorm3( F.relu( self.conv3( x)))
        x = self.pooling2(self.batchNorm4( F.relu( self.conv4( x))))
        #print(x.shape[1]*x.shape[2]*x.shape[3])
        ##x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = x.view(-1, 4096)
        #
        x = F.relu( self.fc1( x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNMNIST1(nn.Module):
    def __init__(self, args):
        super(CNNMNIST1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=args.num_channels,out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pooling1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pooling2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(3136, 512)
        self.drop3 = nn.Dropout2d(p=0.25)
        self.fc3 = nn.Linear(512, 10)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.batchNorm4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.batchNorm1( F.relu( self.conv1( x)))
        x = self.pooling1(self.batchNorm2( F.relu( self.conv2( x))))
        x = self.batchNorm3( F.relu( self.conv3( x)))
        x = self.pooling2(self.batchNorm4( F.relu( self.conv4( x))))
        x = x.view(-1, 3136)#x.shape[1]*x.shape[2]*x.shape[3])
        #x = x.view(3136, -1)
        #
        x = F.relu( self.fc1( x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
'''
#异步联邦学习的模型
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pooling1 = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pooling2 = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(2048, 128)
        self.drop3 = nn.Dropout2d(p=0.25)
        self.fc3 = nn.Linear(128, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.drop1( self.pooling1( F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = self.drop2( self.pooling2( F.relu(self.conv4(x)) ) )

        #print(x.shape[1]*x.shape[2]*x.shape[3])
        ##x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = x.view(-1, 2048)
        #
        x = self.drop3(F.relu( self.fc1(x) ))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
'''
