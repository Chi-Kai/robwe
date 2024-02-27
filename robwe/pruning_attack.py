import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune
from utils.train_utils import  get_model,getdata
from utils.options import args_parser
from utils.watermark import get_X, get_b
from utils.trainer_private import TesterPrivate
from utils.watermark import get_layer_weights_and_predict,compute_BER
from models.test import test_img_local_all


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

  

def validate(args,net,X,b):
    X = torch.tensor(X, dtype=torch.float32).to(args.device)
    b = torch.tensor(b, dtype=torch.float32).to(args.device)
    success_rate = -1
    pred_b = get_layer_weights_and_predict(net,X,args.device)
    success_rate = compute_BER(pred_b,b,args.device)
    return success_rate

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

def main(args,loadpath):
    init_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    model = get_model(args)
    water_path = args.save_path + '/watermark/'+str(args.frac)+'/'+ str(args.embed_dim) +'/'+str(args.epochs)
    save_path = args.save_path + '/dataset/' + str(args.frac) + '_' + str(args.embed_dim) + '_' + str(args.epochs)
    dict_users_train = torch.load(save_path + '_dataset_train.pt')
    dict_users_test = torch.load(save_path + '_dataset_test.pt')
    '''
    dict_X = torch.load(water_path + '/dict_X.pt')
    dict_b = torch.load(water_path + '/dict_b.pt')
    '''
    keys = torch.load(water_path + '/keys.pt')
    
    model_state = torch.load(loadpath,map_location=torch.device('cpu'))
    prunedf = []

    for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 82,84,86,88,90]:
        model.load_state_dict(model_state)
        w_locals = {}
        res = {}
        for i in range(args.num_users):
          if i == 10 : 
              w_local_dict = model.state_dict()
              loaded_dict = np.load( args.save_path + '/head/0.1/{}/{}/11.npy'.format(args.embed_dim,args.epochs), allow_pickle=True).item()
              state_dict = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in loaded_dict.items()}
              for key in state_dict.keys():
                w_local_dict[key] = state_dict[key]
              w_locals[i] = w_local_dict
              continue
          
          w_local_dict = model.state_dict()
          loaded_dict = np.load( args.save_path + '/head/0.1/{}/{}/{}.npy'.format(args.embed_dim,args.epochs,i), allow_pickle=True).item()
          state_dict = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in loaded_dict.items()}
          for key in state_dict.keys():
            w_local_dict[key] = state_dict[key]
          w_locals[i] = w_local_dict
        
        for i in range(args.num_users):

            pruned_model = copy.deepcopy(model)
            pruned_model.load_state_dict(w_locals[i])
            #pruning_resnet(pruned_model, perc)
            #pruning_resnet_layer(pruned_model, perc, ['conv1.weight','conv2.weight'])
            #amount = perc/100
            #prune.random_unstructured(pruned_model.fc3, name="weight", amount=amount)
            #prune.random_unstructured(pruned_model.fc2, name="weight", amount=amount)
            #prune.l1_unstructured(pruned_model.fc3, name="weight", amount=amount)
            #prune.l1_unstructured(pruned_model.fc2, name="weight", amount=amount)
            #prune.ln_structured(pruned_model.fc3, name="weight", amount=amount, n=2, dim=0)
            #prune.remove(pruned_model.fc3, 'weight')
            #prune.remove(pruned_model.fc2, 'weight')
            w_locals[i] = pruned_model.state_dict()
      
        indd = None 
        #w_glob_keys = [model.weight_keys[i] for i in [0, 1, 3, 4]]
        #w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        

        acc_test, loss_test = test_img_local_all(model, args, dataset_test, dict_users_test,
                                                 w_locals=w_locals,indd=indd,
                                                 dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                 return_all=False)
        all_success_rate = []
        for i in range(args.num_users):
         # model.load_state_dict(w_locals[i])
          w_model = copy.deepcopy(model)
          w_model.load_state_dict(w_locals[i])
          if args.use_watermark:
            tester = TesterPrivate(w_model,args.device)
            success_rate = tester.test_signature(keys[i],0)
            all_success_rate.append(success_rate)
          #print(success_rate)

        acc_watermark = sum(all_success_rate) / args.num_users
        acc_test = acc_test / 100
        res['perc'] = perc
        res['acc_watermark'] = acc_watermark
        res['acc_model']     = acc_test 
        print('prec: {:3d}, acc_watermark: {: 3f}, acc_model: {: 3f}'.format(perc,acc_watermark,acc_test))
        prunedf.append(res)
    savepath = args.save_path + '/prunedf/'+str(args.frac)+'/'+str(args.embed_dim)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    df = pd.DataFrame({'perc': [i ['perc'] for i in prunedf],'acc_watermark': [
         i ['acc_watermark'] for i in prunedf],'acc_model': [i ['acc_model'] for i in prunedf]})
    #pd.DataFrame(prunedf).to_csv('./save/prunedf/'+str(args.frac)+'/'+str(args.embed_dim)+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
    #           args.shard_per_user) +'_'+str(args.epochs)+'_'+str(args.perc)+'.csv')
    df.to_csv(savepath+'/'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+str(args.epochs)+'_'+str(perc)+'.csv')


if __name__ == '__main__':

   args = args_parser()
   emds = [0,50,80,100,200,300]
   for emd in emds:
        args.embed_dim = emd
        args.use_watermark = True
        if emd == 0:
            args.use_watermark = False
        args.frac = 0.1
        args.epochs = 50
        model_save_path = args.save_path + "/models/" + str(args.frac)+'/'+str(args.embed_dim)+'/'+str(args.epochs) + '/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_iter'+ str(args.use_watermark) + str(args.epochs) + '.pt'
        main(args,loadpath=model_save_path)

