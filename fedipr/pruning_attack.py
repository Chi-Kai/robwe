import os
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune
from experiments.trainer_private import TesterPrivate
from models.Nets import CNNCifar
from utils.train_utils import getdata
from utils.args import parser_args
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

def pruning_resnet(model, pruning_perc):

    if pruning_perc == 0:
        return
    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()
    
    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for p in model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())

def pruning_resnet_layer(model, pruning_perc,layers):

    if pruning_perc == 0:
        return
    allweights = []
    for name, p in model.named_parameters():
        if name in layers:
            allweights += p.data.cpu().abs().numpy().flatten().tolist()
    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for name, p in model.named_parameters():
        if name in layers:
            mask = p.abs() > threshold
            p.data.mul_(mask.float())


def prune_linear_layer(layer, pruning_method, amount):
  
    pruning_method.apply(layer, 'weight', amount)
    prune.remove(layer, 'weight')

  
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

def main(args,loadpath,water_path):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    train_set,test_set,dict_train = getdata(dataset=args.dataset,
                                                        datadir = args.data_root,
                                                        partition = args.iid,
                                                        num_users = args.num_users
                                                        )
    val_ldr = DataLoader(test_set, batch_size=20, shuffle=False, num_workers=2)
    model = CNNCifar().to(args.device)
    keys = torch.load(water_path)
    keys = keys[0]
    prunedf = []
    
    for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 82,84,86,88,90]:
        accs = []
        res  = {}
        model.load_state_dict(torch.load(loadpath))
        #pruning_resnet_layer(pruned_model, perc, ['fc2.weight','fc3.weight'])
        #pruning_resnet_layer(model, perc,['conv1.bias','conv1.scale','conv2.bias','conv2.scale']) 
        pruning_resnet(model,perc)
        testrer = TesterPrivate(model, args.device)
        for idx in range(args.num_users):
            sign_acc = testrer.test_signature(keys[idx],0)
            if sign_acc != None:
                accs.append(sign_acc)
        acc_test = sum(accs) / len(accs)
        loss,acc = test(model,args.device,val_ldr)
        res['perc'] = perc
        res['acc_watermark'] = acc_test
        res['acc_model']     = acc
        print('prec: {:3d}, acc_watermark: {: 3f}, acc_model: {: 3f}'.format(perc,acc_test,acc))
        prunedf.append(res)
    savepath = 'FedIPRsave/prunedf/'+str(args.frac)+'/'+str(args.embed_dim)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df = pd.DataFrame({'perc': [i ['perc'] for i in prunedf],'acc_watermark': [
         i ['acc_watermark'] for i in prunedf],'acc_model': [i ['acc_model'] for i in prunedf]})
    #pd.DataFrame(prunedf).to_csv('./save/prunedf/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
    #           args.shard_per_user) +'_'+str(args.epochs)+'_'+str(args.perc)+'.csv')
    df.to_csv(savepath+'/'+ str(args.embed_dim)+'_' + str(args.num_users) +'_'+str(perc)+'.csv')


if __name__ == '__main__':

   args = parser_args()
   args.num_users = 10
   emds = [50,80,100,200,300]
   for emd in emds:
        args.embed_dim = emd
        args.use_watermark = True
        if emd == 0:
            args.use_watermark = False
        args.frac = 1
        args.epochs = 50
        model_save_path =  "FedIPRsave/glob_models/" + str(args.frac)+'/'+str(args.epochs)+'/'+ str(emd) + '_' +str(args.num_users) + '.pt'
        water_path = "FedIPRsave/key/" + str(args.frac)+'/'+str(args.epochs) + '/' + str(emd) + '_' +str(args.num_users) + '_key.pt'
        main(args,model_save_path,water_path)

