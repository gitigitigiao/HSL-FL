import sys
import os
from os import path
sys.path.append('/home/workspace/python/HSL-FL' )

import torch
from torchvision import datasets, transforms
from utils.sampling import cifar_iid
from models.test import test_img

def load_dataset_split_user(args):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("runs on "+str(args.device))
    # load dataset and split users
    data_return = dict()
    dict_name = './data/'
    if args.dataset == 'cifar':
        dict_name += 'cifar-'
    else:
        exit('Error: unrecognized dataset')
    
    if args.iid:
        dict_name += 'iid.pkl'
    else:
        dict_name += 'noniid.pkl'

    import os
    import pickle
    if os.path.exists(dict_name) == True:
        with open(dict_name, 'rb') as f:
            obj = pickle.load(f)
            dataset_train = obj["dataset_train"]
            dataset_test = obj["dataset_test"]
            dict_users = obj["dict_users"]
            user_data_value = list()
    else:
        if args.dataset == 'cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
            
            if args.iid == True:
                dict_users = cifar_iid(dataset_train, args.num_users)
            else:
                exit('Error: only consider IID setting in CIFAR10')
        else:
            exit('Error: unrecognized dataset')
        
        data_return["dataset_train"] = dataset_train
        data_return["dataset_test"] = dataset_test
        data_return["dict_users"] = dict_users
        user_data_value = list()
        with open(dict_name, 'wb') as f:
            pickle.dump(data_return, f, pickle.HIGHEST_PROTOCOL)

    img_size = dataset_train[0][0].shape
    return dataset_train ,dataset_test ,dict_users ,user_data_value ,img_size

def reprocessing(net_glob,dataset_train,dataset_test,args,log_lines,epoch_log_lines,algorithm):
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    updateStrategy = ''
    print("({}-{})Training accuracy: {:.2f}".format(algorithm,args.updateStrategy,acc_train))
    print("({}-{})Testing accuracy: {:.2f}".format(algorithm,args.updateStrategy,acc_test))
    log_name, epoch_log_name = "", ""
    if algorithm == 'Fed':
        log_name = './log/{}/(communication)FedAvg_{}_iid{}_{}_epoch={}_userNum={}_tau={}.txt'.format(
            args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep)
        epoch_log_name = './log/{}/(epoch)FedAvg_{}_iid{}_{}_epoch={}_userNum={}_tau={}.txt'.format(
            args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep)
    elif algorithm == 'Random':
        log_name = './log/{}/(communication)FedRandom_{}_iid{}_{}_epoch={}_userNum={}_tau={}.txt'.format(
            args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep)
        epoch_log_name = './log/{}/(epoch)FedRandom_{}_iid{}_{}_epoch={}_userNum={}_tau={}.txt'.format(
            args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep)
    elif algorithm =='Roulette':
        if args.updateStrategy == 'fixed':
            log_name = './log/{}/(communication)FedRoulette_{}_iid{}_{}_epoch={}_userNum={}_fixed_tau={}_{}_T={}_beta={}.txt'.format(
                args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep, args.aggregation,  args.T, args.beta)
            epoch_log_name = './log/{}/(epoch)FedRoulette_{}_iid{}_{}_epoch={}_userNum={}_fixed_tau={}_{}_T={}_beta={}.txt'.format(
                args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep, args.aggregation,  args.T, args.beta)
        else:
            log_name = './log/{}/(communication)FedRoulette_{}_iid{}_{}_epoch={}_userNum={}_adaptive_tau={}_{}_alpha={}.txt'.format(
                args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep, args.aggregation,  args.alpha)
            epoch_log_name = './log/{}/(epoch)FedRoulette_{}_iid{}_{}_epoch={}_userNum={}_adaptive_tau={}_{}_alpha={}.txt'.format(
                args.times, args.dataset, args.iid, args.model, args.epochs, args.select_num, args.local_ep, args.aggregation,  args.alpha)
                
    if os.path.exists(log_name):
        with open(log_name,'w') as log:
            log.write('\n')
            log.write('\n'.join(str(x) for x in log_lines))
    else:
        with open(log_name,'w') as log:
            log.write('\n'.join(str(x) for x in log_lines))

    if os.path.exists(epoch_log_name):
        with open(epoch_log_name,'w') as log:
            log.write('\n')
            log.write('\n'.join(str(x) for x in epoch_log_lines))
    else:
        with open(epoch_log_name,'w') as log:
            log.write('\n'.join(str(x) for x in epoch_log_lines))
    print("Task finished.")