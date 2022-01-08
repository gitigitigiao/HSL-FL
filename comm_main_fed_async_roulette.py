#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import sys
import os
from os import path

from utils.HSL_FL import HSL_FL

sys.path.append('/home/workspace/python/HSL-FL')
import copy
import numpy as np

from utils.options import args_parser
from utils.data_process import load_dataset_split_user, reprocessing
from utils.model_build import model_build

from utils.sleep_time import sleep_user, get_comm_list
from models.Fed import FedRoulette_HS
from models.Update import LocalUpdate
from models.test import test_img
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import time
import math

import torch

# parse args
args = args_parser()
count_number = 0
active_user_list = []


def setWorker(cv_xt, net_glob, user, cur_version, HH):

    cv_xt.acquire()
    global dict_users, flag
    global hsl_fl_model
    global num_workers, count_number, client_net, client_model_version, client_intensity
    global t_train, saved_result, sum_t
    global loss_locals, comm_loss, comm_list
    
    net_local = copy.deepcopy(net_glob)
    cv_xt.release()
    d_idx = dict_users[user]

    tau = time.time()
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=d_idx)
    cpu_sleep_time_per_tau = sleep_user(user)
    
    w_local, loss, gradient_, init_m = local.routtle_train(net=net_local, intensity=HH, delay=cpu_sleep_time_per_tau)

    t = time.time()


    time.sleep(comm_list[user])
    now = time.time()
    comm_time = now - t

    cv_xt.acquire()
    t_train[user] = t - tau
    sum_t += t_train[user]



    client_net[user] = copy.deepcopy(w_local)
    client_model_version[user] = cur_version
    client_intensity[user] = HH
    active_user_list.remove(user)
    sendtime = time.time()
    saved_list = (user, t, tau, w_local, hsl_fl_model.H[user], sendtime, now, cur_version, comm_time, gradient_, init_m)
    saved_result.append(saved_list)
    count_number += 1
    num_workers -= 1

    comm_loss.append(loss)
    loss_locals.append(loss)

    cv_xt.release()

def clientSelectNew(cv_xt, net_glob, num=0):
    if num <= 0:
        return

    global args, sum_H, cur_epoch, num_workers

    ability_pred, set_H_list, ability_rank = dict(), [], []

    for i in range(args.num_users):
        if args.train_ucb and i not in hsl_fl_model.ability.keys():
            ability_pred[i] = float(hsl_fl_model.roulette[i])
            continue

        hsl_fl_model.roulette[i] = hsl_fl_model.get_reward(i)
        ability_rank.append([i, hsl_fl_model.get_ucb_ability_value(i)])

    selected_user_list = []
    Candidate = []
    for i in range(args.num_users):
        if i not in active_user_list:
            Candidate.append(i)
    while len(selected_user_list) < num:
        sum1 = 0.0
        for u in Candidate:
            sum1 += hsl_fl_model.roulette[u]
        shot = random.random() * sum1

        for i in Candidate:
            shot -= hsl_fl_model.roulette[i]
            if shot <= 0:
                user = i
                if user not in active_user_list:
                    active_user_list.append(user)
                    selected_user_list.append(user)
                    Candidate.remove(user)
                break


    avg_delta = 0.0
    for user in selected_user_list:
        avg_delta += hsl_fl_model.get_ucb_ability_value(user)
    avg_delta = avg_delta / len(selected_user_list)

    for user in selected_user_list:
        if args.train_ucb == False or(args.train_ucb == True and user in hsl_fl_model.ability.keys()) :
            ability_pred[user] = hsl_fl_model.get_ucb_ability_value(user)

            if args.updateStrategy == 'fixed':
                hsl_fl_model.set_H_fixed(user, ability_pred[user], args.beta, args.T)
            elif args.updateStrategy == 'adaptive':
                dddd = ability_pred[user] / avg_delta
                hhhhhhh = args.local_ep * dddd
                if hhhhhhh > 0:
                    hsl_fl_model.H[user] = math.ceil(hhhhhhh)
                else:
                    hsl_fl_model.H[user] = 1
        else:
            hsl_fl_model.H[user] = args.local_ep

        sum_H += hsl_fl_model.H[user]
        set_H_list.append(hsl_fl_model.H[user])

    for i in range(len(selected_user_list)):
        thread_w = threading.Thread(target=setWorker, args=(cv_xt, net_glob,
            selected_user_list[i], cur_epoch, set_H_list[i],))
            
        thread_w.start()


    num_workers += num


def despatcher(cv_xt, net_glob):
    global exit_flag, num_workers, _iter, total_iter, last_number, args

    if exit_flag == 1 or exit_flag == 2:
        return 0
    elif _iter >= total_iter:
        exit_flag = 1
        return 0

    num_Client = args.select_num - num_workers
    clientSelectNew(cv_xt, net_glob, num=num_Client)
    _iter = _iter + num_Client
    last_number = num_Client


def globalUpdate(net_glob, saved_result, updateStrategy, deadline):
    print('global update:...')
    global args

    global count_number, init_time
    global exit_flag
    global cur_communication, cur_epoch, old_model
    global saved_glob_comm, saved_cur_comm, saved_time_comm, client_intensity, client_model_version, client_net
    global saved_glob_epoch, saved_time_epoch, saved_intensity_epoch, saved_len_epoch, saved_cur_epoch
    global epoch_avg_loss, loss_locals
    global hsl_fl_model
    w_locals = []
    s_list = []
    H_list = []

    index = 0
    lenth = len(saved_result)

    epoch_total_intensity = 0
    w_gradient_ = []
    init_m_ = []
    for i in range(len(client_intensity)):
        client_intensity[i] = 0
    while index < lenth:
        temp = saved_result[index]
        user, t, tau, w_local, intensity, recieved_time, now, model_version, comm_time, gradient_, init_m = temp[0], \
                                                                                                            temp[1], \
                                                                                                            temp[2], \
                                                                                                            temp[3], \
                                                                                                            temp[4], \
                                                                                                            temp[5], \
                                                                                                            temp[6], \
                                                                                                            temp[7], \
                                                                                                            temp[8], \
                                                                                                            temp[9], \
                                                                                                            temp[10]

        if updateStrategy == 'fixed' and recieved_time > deadline:
            index += 1
            continue

        epoch_total_intensity += intensity
        w_locals.append(copy.deepcopy(w_local))
        w_gradient_.append(copy.deepcopy(gradient_))
        init_m_.append(copy.deepcopy(init_m))
        s_list.append(max(cur_epoch - model_version, 1))
        H_list.append(intensity)


        ability_temp = float(intensity) / (t - tau)
        if user not in hsl_fl_model.ability.keys():
            hsl_fl_model.ability[user] = 0.0

        hsl_fl_model.ucb_ability.update(user, ability_temp)
        hsl_fl_model.ucb_comm.update(user, 5.0 / comm_time)

        cur_communication += 1
        cur_total_time = now - init_time

        client_intensity[user] = intensity

        saved_glob_comm.append(copy.deepcopy(net_glob))
        saved_cur_comm.append(cur_communication)
        saved_time_comm.append(cur_total_time)

        del saved_result[index]
        lenth -= 1

    if w_gradient_:
        w_glob = dict()
        w_glob = FedRoulette_HS(net_glob.state_dict(), w_locals, s_list, H_list)

        net_glob.load_state_dict(w_glob)


        cur_epoch += 1
        cur_total_time = time.time() - init_time

        saved_glob_epoch.append(copy.deepcopy(net_glob))
        saved_time_epoch.append(cur_total_time)
        saved_intensity_epoch.append(epoch_total_intensity)
        saved_len_epoch.append(len(w_locals))
        saved_cur_epoch.append(cur_epoch)
        epoch_avg_loss.append(sum(loss_locals) / len(loss_locals))
        loss_locals = []

    if cur_communication >= total_iter:
        exit_flag = 2

    print('global update finished')
    return net_glob


def updateController(cv_xt, net_glob, updateStrategy, T=0, alpha=0.1, total_workers=100):
    initial_time = time.time()
    time_pre = copy.deepcopy(initial_time)
    global flag, saved_result, args, epoch_remains, exit_flag, last_number
    if updateStrategy == 'adaptive':
        threshold = alpha * args.select_num
        while 1:
            if exit_flag == 2:
                return
            if (len(saved_result) >= int(threshold)) or (exit_flag == 1 and num_workers == 0):
                cv_xt.acquire()
                net_glob = globalUpdate(net_glob, saved_result, updateStrategy, 0)
                despatcher(cv_xt, net_glob)
                cv_xt.release()
            time.sleep(1)
    elif updateStrategy == 'fixed':
        while 1:
            time_now = time.time()
            while time_now - time_pre < T:
                if exit_flag == 2:
                    return
                time.sleep(10)
                time_now = time.time()
                continue
            if exit_flag == 2:
                return
            time_pre = time_now
            
            cv_xt.acquire()
            net_glob = globalUpdate(net_glob, saved_result, updateStrategy, time_pre)
            despatcher(cv_xt, net_glob)
            cv_xt.release()
    else:
        exit('Error: wrong parameter --updateStrategy')


def recordComm(parrr):
    global commOver0, commOver1, communication_log_lines
    global saved_glob_comm, saved_cur_comm, saved_time_comm, dataset_test, dataset_train, args
    global comm_loss, epoch_avg_loss
    eee = int(len(saved_glob_comm) / 2)

    if parrr == 0:
        for i in range(eee):
            if i == 0 or saved_glob_comm[i] != saved_glob_comm[i - 1]:
                cur_acc, cur_loss = test_img(saved_glob_comm[i], dataset_test, args)
                _cur_loss = comm_loss[i]
            communication_log_lines[saved_cur_comm[i] - 1] = str(cur_acc / 100.0) + " " + str(cur_loss) + " " + str(
                saved_cur_comm[i]) + " " + str(saved_time_comm[i]) + " " + str(_cur_loss)
        commOver0 = 1
    else:
        for i in range(eee, len(saved_glob_comm)):
            if i == 0 or saved_glob_comm[i] != saved_glob_comm[i - 1]:
                cur_acc, cur_loss = test_img(saved_glob_comm[i], dataset_test, args)
                _cur_loss = comm_loss[i]
            communication_log_lines[saved_cur_comm[i] - 1] = str(cur_acc / 100.0) + " " + str(cur_loss) + " " + str(
                saved_cur_comm[i]) + " " + str(saved_time_comm[i]) + " " + str(_cur_loss)
        commOver1 = 1

    print(communication_log_lines)

def recordEpoch(parrr):
    global epochOver0, epochOver1, epoch_log_lines
    global saved_glob_epoch, dataset_test, dataset_train, args, saved_cur_epoch, saved_time_epoch, saved_len_epoch, saved_intensity_epoch
    global comm_loss, epoch_avg_loss
    eee = int(len(saved_glob_epoch) / 2)
    print(len(saved_glob_epoch))
    print(eee)
    if parrr == 0:
        for j in range(eee):
            cur_acc, cur_loss = test_img(saved_glob_epoch[j], dataset_test, args)
            _cur_loss = epoch_avg_loss[j]

            epoch_log_lines[saved_cur_epoch[j] - 1] = str(cur_acc / 100.0) + " " + str(cur_loss) + " " + str(
                saved_cur_epoch[j]) + " " + str(saved_time_epoch[j]) + " " + str(saved_len_epoch[j]) + " " + str(
                saved_intensity_epoch[j] / saved_len_epoch[j]) + " " + str(_cur_loss)
        epochOver0 = 1
    else:
        for j in range(eee, len(saved_glob_epoch)):
            cur_acc, cur_loss = test_img(saved_glob_epoch[j], dataset_test, args)
            _cur_loss = epoch_avg_loss[j]

            epoch_log_lines[saved_cur_epoch[j] - 1] = str(cur_acc / 100.0) + " " + str(cur_loss) + " " + str(
                saved_cur_epoch[j]) + " " + str(saved_time_epoch[j]) + " " + str(saved_len_epoch[j]) + " " + str(
                saved_intensity_epoch[j] / saved_len_epoch[j]) + " " + str(_cur_loss)
        epochOver1 = 1
    print(epoch_log_lines)
    epochOver = 1

if __name__ == '__main__':
    cv_xt = threading.Condition()
    saved_glob_comm, saved_cur_comm, saved_time_comm = [], [], []
    saved_glob_epoch, saved_time_epoch, saved_intensity_epoch, saved_len_epoch, saved_cur_epoch = [], [], [], [], []

    epoch_avg_loss = []
    comm_loss = []
    loss_locals = []


    dataset_train, dataset_test, dict_users, user_data_value, img_size = load_dataset_split_user(args)
    net_glob, client_net, client_model_version, client_intensity = model_build(args, algorithm='Roulette')
    w_glob = net_glob.state_dict()

    w = []
    for i in range(args.num_users):
        w.append(10.0 * float(len(w_glob)))

    cur_communication, cur_epoch = 0, 0
    communication_log_lines, epoch_log_lines = [], []

    last_number = 0
    old_model = 0

    total_iter, _iter = args.epochs * 10, 0


    hsl_fl_model = HSL_FL(args.num_users, mu=args.mu, dataset=args.dataset)
    if hsl_fl_model.load_from_file:
        args.train_ucb = False
    else:
        args.train_ucb = True

    saved_result = list()
    t_train = dict()

    flag = 0
    exit_flag = 0

    sum_H, sum_t = 0, 0

    comm_list = get_comm_list(args.num_users)

    total_iter = args.epochs * 10

    init_time = time.time()

    pool = ThreadPoolExecutor(30)

    num_workers = 0

    clientSelectNew(cv_xt, net_glob, num=args.select_num)
    _iter = _iter + args.select_num

    thread_w = threading.Thread(target=updateController, args=(
    cv_xt, net_glob, args.updateStrategy, args.T, args.alpha, args.num_users,))
    thread_w.start()
    while 1:
        if exit_flag == 2:
            break
        time.sleep(10)

    exit_flag = 2


    commOver = 0
    commOver0 = 0
    commOver1 = 0
    epochOver0 = 0
    epochOver1 = 0

    for _ in range(len(saved_glob_comm)):
        communication_log_lines.append("")

    thread_r1 = threading.Thread(target=recordComm, args=(0,))
    thread_r1.start()
    thread_r21 = threading.Thread(target=recordComm, args=(1,))
    thread_r21.start()

    for _ in range(len(saved_glob_epoch)):
        epoch_log_lines.append("")

    thread_e1 = threading.Thread(target=recordEpoch, args=(0,))
    thread_e1.start()
    thread_e21 = threading.Thread(target=recordEpoch, args=(1,))
    thread_e21.start()

    while commOver0 == 0 or commOver1 == 0 or epochOver0 == 0 or epochOver1 == 0:
        time.sleep(5)

    hsl_fl_model.save_ucb()
    reprocessing(net_glob=net_glob, dataset_train=dataset_train, dataset_test=dataset_test, args=args,
                 log_lines=communication_log_lines, epoch_log_lines=epoch_log_lines, algorithm='Roulette')
