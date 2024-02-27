import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset,datedir,idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(datedir, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(datedir, idx)

    if is_train:
        train_data = read_data(datedir, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(datedir, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

#----------------------#
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
        
    elif dataset == 'Cifar10':
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
    dict_users_train,dict_users_test= partition_data(dataset,y_train,y_test,args.partition,args.num_clients,args.beta)
    
    return  dataset_train,dataset_test,dict_users_train,dict_users_test

def partition_data(dataset,train_label,test_label,partition, n_parties, beta=0.4):
    
    n_train = len(train_label)
    n_test  = len(test_label)

    train_label = np.array(train_label)
    test_label  = np.array(test_label)

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test  = np.random.permutation(n_test)
        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)
        train_dataidx_map = {i: batch_idxs_train[i] for i in range(n_parties)}
        test_dataidx_map  = {i: batch_idxs_test[i]  for i in range(n_parties)}

    elif partition == "noniid-labeldir":
        min_size_train = 0
        min_size_test  = 0
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200


        train_dataidx_map = {}
        test_dataidx_map = {}
        train_label_pro = {}

        while min_size_train < min_require_size:
           train_idx_batch = [[] for _ in range(n_parties)]
           for k in range(K):
                idx_k_train = np.where(train_label == k)[0]
                np.random.shuffle(idx_k_train)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                p_train = np.array([p * (len(idx_j) < n_train / n_parties) for p, idx_j in zip(proportions, train_idx_batch)])
                p_train = p_train / p_train.sum()
                p_train = (np.cumsum(p_train) * len(idx_k_train)).astype(int)[:-1]
                train_idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(train_idx_batch, np.split(idx_k_train, p_train))]
                min_size_train = min([len(idx_j) for idx_j in train_idx_batch])
        
        test_idx_by_label = [np.where(test_label == k)[0] for k in range(K)]
        for j in range(n_parties):
            np.random.shuffle(train_idx_batch[j])
            train_dataidx_map[j] = train_idx_batch[j]

            train_label_pro = np.bincount(train_label[train_dataidx_map[j]], minlength=K) / len(train_dataidx_map[j]) 
            n_test_j = int(len(train_dataidx_map[j]) * n_test / n_train)
            test_idx_batch = []
            for k in range(K):
                n_test_j_k = int(n_test_j * train_label_pro[k])
                if len(test_idx_by_label[k]) < n_test_j_k:
                    test_idx_batch.extend(test_idx_by_label[k])
                else:
                    test_idx_batch.extend(np.random.choice(test_idx_by_label[k], n_test_j_k, replace=False))
            test_dataidx_map[j] = test_idx_batch            
            #print('Client {}: {} train, {} test'.format(j, len(train_dataidx_map[j]), len(test_dataidx_map[j])))

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        if num == 10:
            train_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            test_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k_train = np.where(train_label==i)[0]
                idx_k_test  = np.where(test_label==i)[0]
                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)
                split_train = np.array_split(idx_k_train,n_parties)
                split_test = np.array_split(idx_k_test,n_parties)
                for j in range(n_parties):
                    train_dataidx_map[j]=np.append(train_dataidx_map[j],split_train[j])
                    test_dataidx_map[j] = np.append(test_dataidx_map[j],split_test[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                selected_label=1
                while (selected_label<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        selected_label=selected_label+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)

            train_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            test_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

            for i in range(K):
                idx_k_train = np.where(train_label==i)[0]
                idx_k_test  = np.where(test_label==i)[0]
                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)
                split_train = np.array_split(idx_k_train,times[i])
                split_test  = np.array_split(idx_k_test,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        train_dataidx_map[j]=np.append(train_dataidx_map[j],split_train[ids])
                        test_dataidx_map[j]=np.append(test_dataidx_map[j],split_test[ids])
                        ids+=1

    elif partition == "iid-diff-quantity":
        train_idxs = np.random.permutation(n_train)
        test_idxs = np.random.permutation(n_test)
        train_min_size = 0
        test_min_size = 0
        while train_min_size < 10:
            p_train = np.random.dirichlet(np.repeat(beta, n_parties))
            p_train = p_train/p_train.sum()
            train_min_size = np.min(p_train*len(train_idxs))

        while test_min_size < 10:
            p_test = np.random.dirichlet(np.repeat(beta, n_parties))
            p_test = p_test/p_test.sum()
            test_min_size = np.min(p_test*len(test_idxs))
        p_train = (np.cumsum(p_train)*len(train_idxs)).astype(int)[:-1]
        p_test = (np.cumsum(p_test)*len(test_idxs)).astype(int)[:-1]
        train_batch_idxs = np.split(train_idxs,p_train)
        test_batch_idxs = np.split(test_idxs,p_test)
        train_dataidx_map = {i: train_batch_idxs[i] for i in range(n_parties)}
        test_dataidx_map = {i: test_batch_idxs[i] for i in range(n_parties)}
    
    return train_dataidx_map,test_dataidx_map

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label