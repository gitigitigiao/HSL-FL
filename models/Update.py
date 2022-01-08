#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.optimizer.FedProxOptimizer import FedProxOpt

import os
import time

import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.weight_decay = 0.005

    def routtle_train(self, net, intensity, delay=0):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []

        for iter in range(intensity):
            before = copy.deepcopy(net.state_dict())
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if delay > 0:
                time.sleep(delay)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), [], []
