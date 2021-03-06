#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--select_num', type=int, default=10, help="the number of local epochs: E")

    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')

    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_history', type=int, default=10, help='number of history for Roulette select')

    ##new arguments
    parser.add_argument('--H_min', type=int, default=5, help='random seed (default: 1)')
    parser.add_argument('--H_max', type=int, default=15, help='random seed (default: 1)')
    parser.add_argument('--times', type=str, default=2, help='number of trials')
    parser.add_argument('--portion', type=float, default=0.5, help='')
    parser.add_argument('--updateStrategy', type=str, default='adaptive', help='two options:adaptive update,fixed update')
    parser.add_argument('--T', type=int, default=80, help="time interval of fixed update strategy ")
    parser.add_argument('--beta', type=float, default=1.0, help="parameter for setting the intensity")
    parser.add_argument('--targetAcc', type=float, default=99.0, help="")
    parser.add_argument('--alpha', type=float, default=1.0, help="")
    parser.add_argument('--train_ucb', action='store_false', help="whether uniform data or not")
    parser.add_argument('--mu', type=float, default=0.5, help="whether uniform data or not")

    
    args = parser.parse_args()
    return args
