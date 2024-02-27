# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit: Paul Pu Liang

# For MAML (PerFedAvg) implementation, code was adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# credit: Antreas Antoniou

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim 

from utils.watermark import get_layer_weights_and_predict, compute_BER,Signloss
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
#from models.losses.sign_loss import SignLoss

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None,key=None,rep_x=None,rep_b=None,x_l=None,x_i=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs
        
        self.key = key

        self.X = None
        self.b = None
        self.x_l = None
        self.x_i = None
        if args.use_watermark:
            self.X = rep_x.clone().detach().to(dtype=torch.float32, device=self.args.device)
            self.b = rep_b.clone().detach().to(dtype=torch.float32, device=self.args.device)
            self.b = self.b.view(1, -1)
            self.x_l = x_l
            self.x_i = x_i
    def train(self, net, w_glob_keys, last=False,lr=0.01, args=None):
        bias_p = []
        weight_p = []
        # named_parameters()
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        # 设置中local_ep = 5
        # 设置中local_rep_ep = 1
        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
            else:
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0

        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter == head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # rep 水印部分----------------------------------------------------------------------------------
                '''
                rep_loss = 0
                if args.use_watermark:
                    # 得到参数 转化为一维向量
                    para = net.rep_params()
                    # y 是一个一维向量，用来存储参数
                    y = torch.tensor([], dtype=torch.float32).to(self.args.device)
                    #将每个张量转化为一维向量，然后拼接在一起
                    for i in para:
                        y = torch.cat((y, i.view(-1)), 0)
                    y = y.view(1,-1).to(self.args.device)
                    # 根据长度x_l 和 位置x_i 截取y的一部分 
                    start = self.x_i * self.x_l
                    end = start + self.x_l
                    y = y[:, start:end]
                    # 计算损失
                    rep_loss = args.scale * torch.sum(
                        F.binary_cross_entropy(input=torch.sigmoid(torch.matmul(y, self.X.to(self.args.device))), target=self.b.to(self.args.device)))
                # rep 水印部分----------------------------------------------------------------------------------
                '''
                # 这个参数0是scheme，这里我们直接用0
                sign_loss = Signloss(self.key, net, 0, self.args.device).get_loss(layer_type="head")
                (loss + sign_loss).backward()
                '''
                if (iter < head_eps and self.args.alg == 'fedrep') or last:
                   (loss + sign_loss).backward()
                elif iter == head_eps and self.args.alg == 'fedrep' and not last:
                   (loss + rep_loss).backward()
                '''
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())

                if num_updates == self.args.local_updates:
                    done = True
                    break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd,net


def validate(X,b,net,device):
    success_rate = -1
    pred_b = get_layer_weights_and_predict(net,X.to(device),device)
    success_rate = compute_BER(pred_b.to(device),b.to(device),device)
    return success_rate

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate_fine(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None, X=None, b=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.01, args=None, net_glob=None):
        bias_p = []
        weight_p = []
        net.train()
        # named_parameters()
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        # 设置中local_ep = 5
        # 设置中local_rep_ep = 1
        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
            else:
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0

        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter == head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif self.args.alg != 'fedrep':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())

                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

    def validate(self,net):
        success_rate = -1
        pred_b = get_layer_weights_and_predict(net,self.X.cpu().numpy())
        success_rate = compute_BER(pred_b=pred_b,b=self.b.cpu().numpy())
        return success_rate

    def _local_update_noback_fine(self,model,device,dataloader,local_ep, lr):
        
        optimizer = optim.SGD(model.parameters(),
                              lr,
                              momentum=0.9,
                              weight_decay=0.0005) 
                                  
        model.to(device)
        model.train()
        epoch_loss = []
        train_ldr = dataloader

        for epoch in range(local_ep):
            
            loss_meter = 0
            sign_loss_meter = 0 
            acc_meter = 0 
            
            for batch_idx, (x, y) in enumerate(train_ldr):

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = torch.tensor(0.).to(device)
                pred = model(x)
                loss += F.cross_entropy(pred, y)
                acc_meter += accuracy(pred, y)[0].item()
                loss.backward()
                optimizer.step()
                loss_meter += loss.item()
                   

            loss_meter /= len(train_ldr)
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)
        return  np.mean(epoch_loss)