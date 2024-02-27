# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg),
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import copy
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os
from experiments.trainer_private import TrainerPrivate
from experiments.trainer_private import TesterPrivate
from models.Nets import CNNCifar
from utils.datasets import *
from utils.train_utils import getdata
from utils.args import parser_args
from torch.utils.data import DataLoader
import time
import random
import torch.nn.functional as F

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

def get_model(args):
    if args.dataset == "cifar10":
        return CNNCifar()

def test(model,device,dataloader):

        model.to(device)
        model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(device)
                target = target.to(device)
        
                pred = model(data)  # test = 4
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter

def tests(model,args,dataloader,keys):
    accs = []
    tester = TesterPrivate(model, args.device)
    loss,acc = test(model=model,device=args.device,dataloader=dataloader)
    for idx in range(args.num_users):
        private_sign_acc = tester.test_signature(keys[idx], 0)
        if private_sign_acc != None:
            accs.append(private_sign_acc)   
    acc_test = sum(accs) / len(accs)
    return acc,acc_test

 

def main(args,seed,model_path):
    init_seed(seed=seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    lens = np.ones(args.num_users)
    #dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    train_set, test_set,dict_train = getdata(dataset=args.dataset,
                                                        datadir = args.data_root,
                                                        partition = args.iid,
                                                        num_users = args.num_users
                                                        )
    
    train_ldr = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_ldr = DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)
    local_train_ldrs = []
    for i in range(args.num_users):
        local_train_ldr = DataLoader(DatasetSplit(train_set,dict_train[i]), batch_size = args.batch_size,
                                             shuffle=True, num_workers=2)
        local_train_ldrs.append(local_train_ldr)
    model_glob = get_model(args).to(args.device)
    model_glob.train() 
    model_glob.load_state_dict(torch.load(model_path))
    
    accs , water_accs = [],[]
    water_path = args.save_path + '/key/'+str(args.frac)+'/'+str(args.epochs)
    '''
    dict_X = torch.load(water_path + '/dict_X.pt')
    dict_b = torch.load(water_path + '/dict_b.pt')
    '''
    keys = torch.load(water_path +'/'+str(args.embed_dim)+'_'+str(args.num_sign) +'_key.pt')
    keys = keys[0]
    model = copy.deepcopy(model_glob)
    model.to(args.device)
    acc,acc_test = tests(model,args,val_ldr,keys)
    print('Init Test: acc:{},water_acc:{}'.format(acc,acc_test))
    accs.append(acc)
    water_accs.append(acc_test)
    trainer =  TrainerPrivate(model,args.device,args.dp,args.sigma)
    for iter in range(args.fine_epochs):
        model.train()
        local_w,local_loss = trainer._local_update_noback_fine(local_train_ldrs[1],args.local_ep,args.lr)
        model.load_state_dict(local_w)
        acc,water_acc = tests(model,args,val_ldr,keys)
        print('Round: {}, acc:{},water_acc:{}'.format(iter,acc,acc_test))
        accs.append(acc)
        water_accs.append(acc_test)
    save_path = args.save_path + '/fine_tune/'+str(args.frac)+'/'+str(args.epochs)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame({'acc':accs,'water_acc':water_acc})
    df.to_csv(save_path + '/{}_{}_acc.csv'.format(args.embed_dim,args.num_users))


if __name__ == '__main__':

    args = parser_args()
    frac = [0.1]
    embed_dims = [50,80,100,200] #8320
    args.use_watermark = True
    args.epochs = 100
    args.fine_epochs = 25
    args.num_sign = 10
    for f in frac: 
        for embed_dim in embed_dims:
            args.use_watermark = True
            args.frac = f
            if embed_dim == 0:
                args.use_watermark = False
            args.embed_dim = embed_dim
            model_path = args.save_path + '/glob_models/' + str(args.frac)+'/'+str(args.epochs) +'/'+str(embed_dim)+'_'+str(args.num_users)+'.pt'
            main(args=args,seed=1,model_path=model_path)


    #plot_accs(args,embed_dims,frac,args.epochs)
    #plot_heatmap(args,embed_dims,args.epochs)
    
