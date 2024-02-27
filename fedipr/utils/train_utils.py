# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
from math import *
from utils.sampling import cifar_beta
from utils.sampling import cifar_iid
from utils.utils import *

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

def getdata(dataset,datadir,partition,num_users,beta=0.1):
    dataset = dataset
    datadir = datadir

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
 
    if partition == 'iid':
       dict_users_train = cifar_iid(dataset_train, num_users)
    elif partition == 'noniid':
        dict_users_train = cifar_beta(dataset_train, beta,num_users)
        data_dict = {i: subset for i, subset in enumerate(dict_users_train)}
    else:     
        y_train = dataset_train.targets
        y_test = dataset_test.targets
        dict_users_train,dict_users_test= partition_data(dataset,y_train,y_test,partition,num_users,beta)
        if partition == 'noniid-labeldir':
            for i in range(num_users):
                dict_users_train[i]= set(dict_users_train[i])
        else: 
            for i in range(num_users):
                dict_users_train[i]= set(dict_users_train[i].tolist())
    return  dataset_train,dataset_test,dict_users_train