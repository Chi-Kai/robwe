import os
import time

import numpy as np
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import getdata,DatasetSplit
from utils.watermark import test_watermark,tocsv,get_key

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # create dataset 
        self.dataset_train,self.dataset_test,self.dict_train,self.dict_test = getdata(args)
        self.is_watermark = args.use_watermark and args.watermark_bits > 0
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.test_accs = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(self.dataset_test,self.dict_test,self.dataset_train,self.dict_train,self.test_accs)
                if self.is_watermark:
                    self.watermark_metrics()

            if i == self.global_rounds:
                if self.is_watermark:
                    self.watermark_for_allclients()
                    self.save_client_key()
                self.save_client_model()
                self.save_acc()
          
            for client in self.selected_clients:
                client.train(self.dataset_train,self.dict_train[client.id][:500])

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.test_accs))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    #重载函数
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = DatasetSplit(self.dataset_train,self.dict_train[i][:500])
            test_data = DatasetSplit(self.dataset_test,self.dict_test[i][:500])
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    def test_metrics(self,dataset,dict_test):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics(dataset,dict_test[c.id][:500])
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self,dataset,dict_train):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(dataset,dict_train[c.id][:500])
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, test_dataset,dict_test,train_dataset,dict_train,acc=None, loss=None):
        stats = self.test_metrics(test_dataset,dict_test)
        stats_train = self.train_metrics(train_dataset,dict_train)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
           
    def watermark_metrics(self):
        watermark_accs = []
        for c in self.selected_clients:
            w_acc = test_watermark(c.model,c.key['x'],c.key['b'],self.device,"Conv2d")
            watermark_accs.append(w_acc)
        print("Averaged Watermark Accurancy: {:.4f}".format(np.mean(watermark_accs)))
        #return np.mean(watermark_accs)
    
    # 所有客户端的水印相互测试
    def watermark_for_allclients(self):
        all_client_acc = []
        for ckey in self.clients:
            one_key_acc = []
            for cmodel in self.clients:
                w_acc = test_watermark(cmodel.model,ckey.key['x'],ckey.key['b'],self.device,"Conv2d")
                one_key_acc.append(w_acc)
            all_client_acc.append(one_key_acc)
        tocsv(self.save_folder_name + '/watermark/bits' + str(self.args.watermark_bits)+'_clients'+str(self.args.num_clients)+'.csv',all_client_acc)
    
    def save_client_model(self):
        model_path = os.path.join(self.save_folder_name,"models/"+self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for c in self.clients:
            file_path = model_path +'/' + 'bits'+ str(self.args.watermark_bits)+ "_" + str(c.id) + ".pt"
            torch.save(c.model.state_dict(), file_path)
            
    def save_client_key(self):
        model_path = os.path.join(self.save_folder_name,"keys/"+self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for c in self.clients:
            file_path = model_path + '/' + 'bits'+ str(self.args.watermark_bits)+ "_" + str(c.id) + ".pt"
            torch.save(c.key, file_path)
    
    def save_acc(self):
        model_path = os.path.join(self.save_folder_name,"accs/"+self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        test_acc_dict = {'epochs': range(len(self.test_accs)), 'accs': self.test_accs}
        tocsv(model_path+'/bits'+str(self.args.watermark_bits)+'_clients'+str(self.num_clients)+'.csv',test_acc_dict)
