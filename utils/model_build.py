import sys
from os import path
sys.path.append('/home/workspace/python/HSL-FL' )
import copy

from models.Nets import MLP, CNNMnist, CNNCifar, CNNCIFAR1, CNNMNIST1

def model_build(args,algorithm):
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCIFAR1(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    
    client_net = []
    client_model_version = []
    client_intensity = []
    model_para = net_glob.state_dict()
    for i in range(args.num_users):
        client_net.append(copy.deepcopy(model_para))
        client_model_version.append(0)
        client_intensity.append(10)

    return net_glob, client_net, client_model_version, client_intensity


def modelUpdate():
    return 0