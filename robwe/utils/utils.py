import numpy as np
import torchvision.transforms as transforms
import random
import torch 
def construct_passport_kwargs(self):
    passport_settings = self.passport_config
    model = self.model
    bit_length = self.embed_dim

    passport_kwargs = {}
    
    alexnet_channels = {
        '4': (384, 3456),
        '5': (256, 2304),
        '6': (256, 2304)
    }
    resnet_channels = {
        'layer2': (128, 1152),
        'layer3': (256, 2304),
        'layer4': (512, 4608)
    }

    cnn_channels = {
        '0': (64, 1600),
        '2': (64, 1600),
    }

    for layer_key in passport_settings:
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:  # i = 0, 1 
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]: # module_key = convbnrelu
                    flag = passport_settings[layer_key][i][module_key] # flag = str in module_key
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True

                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag
                    }
                           
 #                  b = torch.sign(torch.rand(self.num_bit) - 0.5)
                    if b is not None:
                        
                        output_channels = int (bit_length * 512 / 2048)
                        output_channels = int (bit_length * 512 / 2048)
  
                        bsign = torch.sign(torch.rand(output_channels) - 0.5)
                        # bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])
                        # for j, c in enumerate(bitstring):
                        #     if c == '0':
                        #         bsign[j] = -1
                        #     else:
                        #         bsign[j] = 1
                        b = bsign

                        if self.weight_type == 'gamma': 
                            M = torch.randn(resnet_channels[layer_key][0], output_channels)
                        else:
                            M = torch.randn(resnet_channels[layer_key][1], output_channels)

                        passport_kwargs[layer_key][i][module_key]['b'] = b
                        passport_kwargs[layer_key][i][module_key]['M'] = M

        else:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            passport_kwargs[layer_key] = {
                'flag': flag
            }
            
            if b is not None:
                if model == 'alexnet':
                    if layer_key == "4":
                        output_channels = int (bit_length * 384 / 896)
                    if layer_key == "5":
                        output_channels = int (bit_length * 256/ 896)
                    if layer_key == "6":
                        output_channels = int (bit_length * 256/ 896)
                if model == 'cnn':
                    if layer_key == "0":
                        output_channels = int (bit_length / 2)
                    if layer_key == "2":
                        output_channels = int (bit_length / 2)                        
                bsign = torch.sign(torch.rand(output_channels) - 0.5)
                # bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])
                              
                # for j, c in enumerate(bitstring):
                #     if c == '0':
                #         bsign[j] = -1
                #     else:
                #         bsign[j] = 1
                b = bsign
                if model == 'alexnet':
                    if self.weight_type == 'gamma': 
                        M = torch.randn(alexnet_channels[layer_key][0], output_channels)
                    else:
                        M = torch.randn(alexnet_channels[layer_key][1], output_channels)
                if model == 'cnn':
                    if self.weight_type == 'gamma': 
                        M = torch.randn(cnn_channels[layer_key][0], output_channels)
                    else:
                        M = torch.randn(cnn_channels[layer_key][1], output_channels) 

                passport_kwargs[layer_key]['b'] = b
                passport_kwargs[layer_key]['M'] = M

    return passport_kwargs

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
