#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedRoulette_HS_avg(w_locals,s_list,H_list):
    w_avg = copy.deepcopy(w_locals[0])
    weight = []
    all_weight = 0
    for i in range(len(w_locals)):
        weight.append( H_list[i] / s_list[i])
        all_weight = all_weight + weight[i]
    for k in w_avg.keys():
        w_avg[k] =  (weight[0]/all_weight)*w_avg[k]
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] = w_avg[k] + (weight[i]/all_weight)*w_locals[i][k]
    return w_avg

def FedRoulette_HS(w_glob, w_locals, s_list, H_list):
    w_temp = FedRoulette_HS_avg(w_locals, s_list, H_list)
    n = len(w_locals)
    theta = 1.0/(n+1)
    w_avg = copy.deepcopy(w_glob)
    for k in w_avg.keys():
        w_avg[k] = theta * w_avg[k] + (1.0 - theta)*w_temp[k]
    return w_avg
