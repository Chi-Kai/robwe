
from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP, CNN_FEMNIST,CNNMnist,VGG16
from models.alexnet import AlexNet
from models.resnet import ResNet18
from utils.sampling import noniid
import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from math import *
import random
from utils.utils import *
import numpy as np
import torch

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def getdata(args):
    dataset = args.dataset
    datadir = args.datadir

    if dataset == 'mnist':
        dataset_train = datasets.MNIST(datadir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(datadir, train=False, download=True, transform=trans_mnist)
        
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(datadir, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(datadir, train=False, download=True, transform=trans_cifar10_val)
        #X_train, y_train, X_test, y_test= load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(datadir, train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100(datadir, train=False, download=True, transform=trans_cifar100_val)
    elif dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST(datadir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST(datadir, train=False, download=True, transform=trans_mnist)
    else:
        exit('Error: unrecognized dataset')
    # sample users
    # dict_users_train 
    y_train = dataset_train.targets
    y_test = dataset_test.targets
    dict_users_train,dict_users_test= partition_data(dataset,y_train,y_test,args.partition,args.num_users,args.beta)
    
    return  dataset_train,dataset_test,dict_users_train,dict_users_test

def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        # dict_users_train
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model(args):
    print("model: {}, dataset: {}".format(args.model, args.dataset))
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset in ['mnist', 'fmnist']:
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'alexnet' and 'cifar10' in args.dataset:
        net_glob = AlexNet(3,10,args.passport_config).to(args.device)
    elif args.model == 'alexnet' and 'cifar100' in args.dataset:
        net_glob = AlexNet(3,100,args.passport_config).to(args.device)
    elif args.model == 'vgg' and 'cifar100' in args.dataset:
        net_glob = VGG16().to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)

    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob
