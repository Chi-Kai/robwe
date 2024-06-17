import csv
import os

import pandas as pd
from models.Nets import CNNCifar,CNNCifar100,CNNMnist,VGG16
from utils.train_utils import getdata
from utils.args import parser_args
# from utils.help import *
from utils.datasets import *
import copy
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import torch.optim as optim

from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from experiments.utils import construct_passport_kwargs,to_csv
from models.alexnet import AlexNet
from models.layers.conv2d import ConvBlock
from models.resnet import ResNet18

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

def watermark_forgery(key, frac):
    key_copy = copy.deepcopy(key)
    b = key_copy['b'].clone()
    s = frac * len(b)
    l = random.sample(range(len(b)), int(s))
    for i in l:
        b[i] = -b[i]
    key_copy['b'] = b
    return key_copy

class IPRFederatedLearning(Experiment):
    """
    Perform federated learning
    """
    def __init__(self, args):
        super().__init__(args) # define many self attributes from args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3
        
        self.num_bit = args.num_bit
        self.num_trigger = args.num_trigger
        self.dp = args.dp
        self.sigma = args.sigma
        self.malice_client_nums = args.malicious_frac * args.num_sign
        self.tampered_frac = args.tampered_frac
        data_root = 'data'
        self.malice_clients = []
        print('==> Preparing data...')
        self.train_set, self.test_set, self.dict_train = getdata(dataset=self.dataset,
                                                        datadir = self.data_root,
                                                        partition = self.iid,
                                                        num_users = self.num_users,
                                                        beta = args.beta
                                                        )
        '''
        with open('./noniid_dir.txt', 'w') as f:
            for key, value in self.dict_train.items():
                f.write('%s:%s\n' % (key, value))
        '''
        print('==> Preparing watermark..')
        if args.backdoor_indis and args.num_back > 0:
            if args.dataset == 'cifar10':
                self.wm_data, self.wm_dict = prepare_wm_new(data_root + 'trigger/cifar10/', self.num_back, self.num_trigger)
            if args.dataset == 'cifar100':
                self.wm_data, self.wm_dict = prepare_wm_indistribution(data_root + 'trigger/cifar100/', self.num_back, self.num_trigger)
        elif args.num_back > 0:
            self.wm_data, self.wm_dict = prepare_wm(data_root + 'trigger/pics', self.num_back)
        

        if self.weight_type == 'gamma':
            if self.loss_type == 'sign':
                self.scheme = 0
            if self.loss_type == 'CE':
                self.scheme = 1
        
        if self.weight_type == 'kernel':
            if self.loss_type == 'sign':
                self.scheme = 2
            if self.loss_type == 'CE':
                self.scheme = 3 

        print('==> Preparing model...')

        self.logs = {'train_acc': [], 'train_sign_acc':[], 'train_wm_acc': [], 'train_loss': [], 'train_sign_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'keys':[],
                     #'trigger_dict': self.wm_dict,

                     'best_test_acc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }

        self.generate_attack_signature_dict()
        self.construct_model()
        
        self.w_t = copy.deepcopy(self.model.state_dict())

        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma)
        self.tester = TesterPrivate(self.model, self.device)

    
    def generate_signature_dict(self):
        
        l = []
        for i in range(self.num_users):
            if i < self.num_sign:
                l.append(1)
            else:
                l.append(0)
        
        np.random.shuffle(l)
        self.keys = []       
        for i in range(self.num_users):
            if l[i] == 1:
                key = construct_passport_kwargs(self)
                self.keys.append(key)
            if l[i] == 0:
                self.keys.append(None)

        self.logs['keys'].append(self.keys)

    def generate_attack_signature_dict(self):
        
        l = []
        for i in range(self.num_users):
            if i < self.malice_client_nums:
                l.append(1)
            elif i < args.num_sign:
                l.append(2)
            else:
                l.append(0)
           
        np.random.shuffle(l)
        self.keys = []   
        key1 = construct_passport_kwargs(self)
        for i in range(args.num_users):
            if l[i] == 1:
                self.keys.append(key1)
                self.malice_clients.append(i)
            if l[i] == 2:
                key2 = construct_passport_kwargs(self)
                self.keys.append(key2)
            if l[i] == 0:
                self.keys.append(None)

        self.logs['keys'].append(self.keys)
             
    def construct_model(self):

        self.passport_kwargs = construct_passport_kwargs(self)

        if self.model_name == 'cnn' and self.dataset == 'cifar10':
            model = CNNCifar(args)
        elif self.model_name == 'cnn' and self.dataset == 'cifar100':
            model = CNNCifar100()
        elif self.model_name == 'cnn' and self.dataset == 'mnist':
            model = CNNMnist(args)
        elif self.model_name == 'cnn' and self.dataset == 'fmnist':
            model = CNNMnist(args)
        elif self.model_name == 'vgg':
            model = VGG16(100)
        else:
            model = ResNet18( passport_kwargs = self.passport_kwargs)
        
        self.model = model.to(self.device)

    def train(self):
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)
        #test_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)
        #wm_test_ldr = DataLoader(self.wm_data, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)

        local_train_ldrs = []
        wm_loaders = []
        for i in range(self.num_users):
            local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_train[i]), batch_size = self.batch_size,
                                             shuffle=True, num_workers=2)
            local_train_ldrs.append(local_train_ldr)

        for i in range(self.num_back):
            # print(self.wm_dict[i])
            # print(len(self.wm_dict[i]))
            wm_loader = DataLoader(DatasetSplit(self.wm_data, self.wm_dict[i]), batch_size=2, shuffle=True, num_workers =2, drop_last=True)
            wm_loaders.append(wm_loader)


        for epoch in range(self.epochs):
            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)

            local_ws, local_losses, sign_losses, private_sign_acces, acc_wms = [], [], [], [], []

            start = time.time()
            for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (self.epochs, self.lr)):
                #self.model.load_state_dict(self.w_t)
                if idx < self.num_back:
                    local_w, local_loss, sign_loss = self.trainer._local_update(local_train_ldrs[idx], wm_loaders[idx], self.local_ep, self.lr, self.keys[idx], self.scheme)
                else:
                    wm_loader = None
                    local_w, local_loss, sign_loss = self.trainer._local_update_noback(local_train_ldrs[idx], wm_loader, self.local_ep, self.lr, self.keys[idx], self.scheme)
    
                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)
                sign_losses.append(sign_loss)

            #self.lr = self.lr * 0.99

            #client_weights = np.ones(self.m) / self.m
            client_weights = []
            sum_len = sum(len(DatasetSplit(self.train_set, self.dict_train[i])) for i in idxs_users)
            for i in idxs_users:
                client_weight = len(DatasetSplit(self.train_set, self.dict_train[i]))/sum_len
                client_weights.append(client_weight)
            
            self._fed_avg(local_ws, client_weights, 1)
            self.model.load_state_dict(self.w_t)
            end = time.time()
            interval_time = end - start
            
            if (epoch + 1) % 1 == 0:

                loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean

                if epoch == self.epochs - 1:           
                    path = args.save_path + '/glob_models/' + str(self.frac) + '/' +str(self.epochs) + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    model_save_path = path +str(self.num_bit)+'_'+str(self.num_users)+'.pt'
                    torch.save(self.model.state_dict(), model_save_path)
                    # test for watermarks
                    if self.num_back>0:
                        for i in range(self.num_back):
                            wm_loader = DataLoader(DatasetSplit(self.wm_data, self.wm_dict[i]), batch_size=2, shuffle=True, num_workers =2, drop_last=True)
                            loss_wm, acc_wm = self.trainer.test(wm_loader)
                            acc_wms.append(acc_wm)
                    else: 
                        acc_wm = 0
                    #恶意检测
                if  self.args.malicious_frac > 0:
                    malicious_accs = []
                    normal_accs = []
                    for idx in range(self.num_users):
                        if idx in self.malice_clients:
                            malicious_acc = self.tester.test_signature(self.keys[idx], self.scheme)
                            if malicious_acc != None:
                                malicious_accs.append(malicious_acc)
                        else:
                            normal_acc = self.tester.test_signature(self.keys[idx], self.scheme)
                            if normal_acc != None:
                                normal_accs.append(normal_acc)
                    if self.malice_client_nums == 0:
                        mean_malicious_acc = 0
                    else:
                        mean_malicious_acc = sum(malicious_accs)/len(malicious_accs)
                    mean_normal_acc = sum(normal_accs)/len(normal_accs)
                    model_path = os.path.join(args.save_path,"sign_accs/"+args.dataset)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    test_acc_dict = {'epochs': range(self.epochs), 'mean_malicious_acc': mean_malicious_acc, 'mean_normal_acc': mean_normal_acc}
                    to_csv(model_path+'/bits'+str(args.num_bit)+'_clients'+str(args.num_users)+'_'+str(self.malice_client_nums)+'_'+str(self.tampered_frac)+'.csv',test_acc_dict)


                    for idx in range(self.num_users):
                        private_sign_acc = self.tester.test_signature(self.keys[idx], self.scheme)
                        if private_sign_acc != None:
                            private_sign_acces.append(private_sign_acc)
                        
                    print('sign_acc: ',private_sign_acces)

                    sign_dir = args.save_path + '/sign_acc/'+str(self.frac)+'/' + str(self.epochs) + '/'
                    if not os.path.exists(sign_dir):
                        os.makedirs(sign_dir)
                    sign_path = sign_dir + str(self.scheme) +'_'+str(self.num_bit)+'_'+str(self.num_users)+'_sign_acc.csv'
                    acc = np.array(private_sign_acces)
                    df = pd.DataFrame({'userid':range(len(acc)), 'acc': acc})
                    df.to_csv(sign_path, index=False)

                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['train_sign_acc'].append(private_sign_acces) 
                #self.logs['train_wm_acc'].append(acc_wm)

                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))


                # use validation set as test set
                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                print('Epoch {}/{}  --time {:.1f}'.format(
                    epoch, self.epochs,
                    interval_time
                )
                )
                mean_sign_loss = 0.
                mean_private_sign_acc = 0.
                mean_acc_wm = 0.

                for i in sign_losses:
                    mean_sign_loss += i
                mean_sign_loss /= len(sign_losses)

                for i in acc_wms:
                    mean_acc_wm += i
                if self.num_back != 0:
                    mean_acc_wm /= self.num_back
                for i in private_sign_acces[:-3]:
                    if i != None:
                        mean_private_sign_acc += i
                
                if self.num_sign != 0:
                    mean_private_sign_acc /= self.num_sign

                print(
                    "Train Loss {:.4f} --- Val Loss {:.4f} --- Sign Loss {:.4f} --- Private Sign Acc {:.4f} -- Watermark Acc {:.4f}"
                    .format(loss_train_mean, loss_val_mean, mean_sign_loss, mean_private_sign_acc, mean_acc_wm))
                print("Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(acc_train_mean, acc_val_mean,
                                                                                                        self.logs[
                                                                                                            'best_test_acc']
                                                                                                        )
                      )
        # if self.num_back>0:
        #     loss_wm, acc_wm = self.trainer.test(wm_test_ldr)
   
        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f} --- Watermark acc{:.4f} --- Sign acc{:.4f}'.format(self.logs['best_test_loss'], 
                                                                                       self.logs['best_test_acc'],
                                                                                        acc_wm, mean_private_sign_acc))
        sign_dir = args.save_path + '/round_acc/'+str(self.frac)+'/' + str(self.epochs) + '/'
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        sign_path = sign_dir + str(self.scheme) +'_'+str(self.num_bit)+'_'+str(self.num_users)+'_round_acc.csv'
        train_acc = np.array(self.logs['train_acc'])
        test_acc  = np.array(self.logs['val_acc'])
        df = pd.DataFrame({'round':range(len(train_acc)), 'train_acc': train_acc, 'test_acc': test_acc})
        df.to_csv(sign_path, index=False)

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean, acc_wm, mean_private_sign_acc

    def _fed_avg(self, local_ws, client_weights, lr_outer):

        w_avg = copy.deepcopy(local_ws[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i]

            self.w_t[k] = w_avg[k]

def tocsv(dir,num_bit,num_sign,acc):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_path=dir+str(num_sign)+'.csv'
    data = {'bits':num_bit, 'acc':acc}
    # Check if file exists
    if os.path.exists(save_path):
        # File exists, open and append new data
        with open(save_path, 'a', newline='') as csvfile:
            fieldnames = ['bits', 'acc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write each row in data to the CSV file
            writer.writerow(data)
    else:
        #File does not exist, create new file and write header row
        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['bits', 'acc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header row to the CSV file
            writer.writeheader()
            writer.writerow(data)

def main(args):
    init_seed(1249)
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,      
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,                
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'num_users': args.num_users
            }
            }

    fl = IPRFederatedLearning(args)

    logg, time, best_test_acc, test_acc, acc_wm, acc_sign = fl.train()                                         
                                             
    logs['net_info'] = logg
    logs['test_acc'] = test_acc
    logs['bp_local'] = True if args.bp_interval == 0 else False

    sign_acc_dir = args.save_path + '/mean_acc_sign/'+str(args.frac)+'/'+str(args.epochs)+'/'
    test_acc_dir = args.save_path + '/best_acc/'+str(args.frac)+'/'+str(args.epochs)+'/'
    key_dir = args.save_path + '/key/'+str(args.frac)+'/'+str(args.epochs)+'/'
    if not os.path.exists(key_dir):
        os.makedirs(key_dir)
    key_path = key_dir + str(args.num_bit)+'_'+str(args.num_sign)+'_key.pt'
    torch.save(logg['keys'], key_path)


    tocsv(sign_acc_dir,args.num_bit,args.num_sign,acc_sign)
    tocsv(test_acc_dir,args.num_bit,args.num_sign,test_acc)
       
 
    return

if __name__ == '__main__':
    args = parser_args()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    frac = [0.1]
    bits = [0,50,100,150] 
    for j in bits:
            args.num_sign = 100
            args.num_bit = j
            print('args.num_bit: {} noniid: {}'.format(args.num_bit,args.iid))
            if j == 0:
                args.num_sign = 0
            main(args)
