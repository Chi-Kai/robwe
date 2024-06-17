import copy
import itertools
import numpy as np
import pandas as pd
import torch
import os
from models.Update import LocalUpdate_fine
from models.test import test_img_local
from torch.utils.data import Dataset, DataLoader
from utils.options import args_parser
from utils.train_utils import get_model,getdata
from utils.trainer_private import TesterPrivate
import torch.nn.functional as F
import random
from utils.watermark import test_watermark
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

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

def main(args,seed,model_path):
    init_seed(seed=seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    save_path = args.save_path + '/dataset/' + str(args.frac) + '_' + str(args.embed_dim) + '_' + str(args.epochs)
    dict_users_train = torch.load(save_path + '_dataset_train.pt')
    dict_users_test = torch.load(save_path + '_dataset_test.pt')
    model_glob = get_model(args).to(args.device)
    model_glob.train() 
    model_glob.load_state_dict(torch.load(model_path))

    total_num_layers = len(model_glob.state_dict().keys())
    print('net_glob.state_dict().keys():')
    print(model_glob.state_dict().keys())
    model_keys = [*model_glob.state_dict().keys()]

    layer_base = []
    if args.alg == 'fedrep' or args.alg == 'fedper':
        layer_base = [model_glob.weight_keys[i] for i in [0,1,2]]
    if args.alg == 'fedavg':
        layer_base = []
    layer_base = list(itertools.chain.from_iterable(layer_base))

    print('total_num_layers:{}'.format(total_num_layers))
    print('w_glob_keys:{}'.format(layer_base)) 
    print('net_keys:{}'.format(model_keys)) 
    print("learning rate:{}, batch size:{}".format(args.lr, args.local_bs))

    model_clients = {}
    for i in [1,20]:
          w_local_dict = model_glob.state_dict()
          loaded_dict = np.load(args.save_path + '/head/0.1/{}/{}/{}.npy'.format(args.embed_dim,args.epochs,i), allow_pickle=True).item()
          state_dict = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in loaded_dict.items()}
          for key in state_dict.keys():
            w_local_dict[key] = state_dict[key]
          model_clients[i] = w_local_dict 
    water_path = args.save_path + '/watermark/'+str(args.frac)+'/'+ str(args.embed_dim) +'/'+str(args.epochs)
    '''
    dict_X = torch.load(water_path + '/dict_X.pt')
    dict_b = torch.load(water_path + '/dict_b.pt')
    '''
    keys = torch.load(water_path + '/keys.pt')

    model = copy.deepcopy(model_glob)
    model.load_state_dict(model_clients[20])
    model.to(args.device)
    # training
    accs_1 = []
    accs_22 = []
    water_acc = []
    #acc_1, loss_1 = test_img_local(model, dataset_test, args, user_idx=1, idxs=dict_users_test[1])
    #acc_22, loss_22 = test_img_local(model, dataset_test, args, user_idx=22, idxs=dict_users_test[22])
    test_ldr_1 = DataLoader(DatasetSplit(dataset_test,dict_users_test[1]), batch_size = args.bs,
                                             shuffle=True, num_workers=2) 
    test_ldr_22 = DataLoader(DatasetSplit(dataset_test,dict_users_test[20]), batch_size = args.bs,
                                             shuffle=True, num_workers=2)
    _,acc_1 = test(model,args.device,test_ldr_1)
    _,acc_22 = test(model,args.device,test_ldr_22)
    acc_1 = acc_1 / 100
    acc_22 = acc_22 / 100
    tester = TesterPrivate(model,args.device)
    #success_rate = tester.test_signature(keys[22],0)    
    if args.use_watermark:              
        success_rate = test_watermark(
                        model,
                        keys[20]["x"],
                        keys[20]["b"],
                        0,
                        0,
                        args.device,
                        "head"
                    )
    print('Init test: acc_1:{},acc_22:{},success_rate:{}'.format(acc_1,acc_22,success_rate))

    train_ldr_1 = DataLoader(DatasetSplit(dataset_train,dict_users_train[1]), batch_size = args.bs,
                                             shuffle=True, num_workers=2)
    train_ldr_22 = DataLoader(DatasetSplit(dataset_train,dict_users_train[20]), batch_size = args.bs,
                                             shuffle=True, num_workers=2)
    for iter in range(args.fine_epochs):
        model.train()
        client = LocalUpdate_fine(args=args, dataset=dataset_train, idxs=dict_users_train[1])
        loss= client._local_update_noback_fine(model,args.device,train_ldr_1,10,args.lr)
        acc_1, loss_1 = test_img_local(model, dataset_test, args, user_idx=1, idxs=dict_users_test[1]) 
        acc_22, loss_22 = test_img_local(model, dataset_test, args, user_idx=20, idxs=dict_users_test[20]) 
        #_,acc_1 = test(model,args.device,test_ldr_1)
        #_,acc_22 = test(model,args.device,test_ldr_22)
        # loss
        #print(loss_22)
        #print(len(dict_users_test[22]))
        acc_1 = acc_1 / 100
        acc_22 = acc_22 / 100
        accs_1.append(acc_1)
        accs_22.append(acc_22)

        #tester = TesterPrivate(model,args.device)
        #success_rate = tester.test_signature(keys[22],0)

        if args.use_watermark:
            success_rate=0
            success_rate = test_watermark(
                    model,
                    keys[20]["x"],
                    keys[20]["b"],
                    0,
                    0,
                    args.device,
                    "head"
                )
            success_rate2 = test_watermark(
            model,
            keys[1]["x"],
            keys[1]["b"],
            0,
            0,
            args.device,
            "head"
        )
        print('Round {:3d}, loss {:.3f}, acc_1 {:.2f}, acc_22 {:.2f}, success_rate {:.2f},success_rate2 {:.2f}'.format(iter, loss, acc_1, acc_22, success_rate,success_rate2))
        water_acc.append(success_rate)
    save_path = args.save_path + '/fine_tune_20/'+str(args.frac)+'/'+str(args.epochs)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame({'accs_1':accs_1,'accs_22':accs_22,'water_acc':water_acc})
    df.to_csv(save_path + '/{}_{}acc.csv'.format(args.embed_dim,args.num_users))


if __name__ == '__main__':

    args = args_parser()
    frac = [0.1]
    embed_dims = [100] #8320
    args.use_watermark = True
    args.epochs = 50
    args.fine_epochs = 25
    for f in frac: 
        for embed_dim in embed_dims:
            args.use_watermark = True
            args.frac = f
            if embed_dim == 0:
                args.use_watermark = False
            args.embed_dim = embed_dim
            model_path = args.save_path + '/models/' + str(args.frac)+'/'+ str(args.embed_dim)+'/'+str(args.epochs) +'/accs_fedrep_cifar10_100_iterTrue'+str(args.epochs)+'.pt'
            main(args=args,seed=args.seed,model_path=model_path)


    #plot_accs(args,embed_dims,frac,args.epochs)
    #plot_heatmap(args,embed_dims,args.epochs)
    
