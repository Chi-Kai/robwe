import os
import random
import time

import numpy as np
import torch
from flcore.clients.clientrep import clientRep
from flcore.servers.serverbase import Server
from threading import Thread
from utils.watermark import test_watermark,tocsv,get_key
from utils.data_utils import read_client_data
class FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_malignant_clients()
        self.set_clients(clientRep)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.test_accs = []
        self.malicious_frac = args.malicious_frac
        self.malignant_clients = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate(acc=self.test_accs)
                self.watermark_metrics()

            if i == self.global_rounds:
                self.watermark_for_allclients()
                self.save_client_model()
                self.save_client_key()
                self.save_acc()

            for client in self.selected_clients:
                client.train()

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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRep)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples 
    
    def watermark_metrics(self):
        watermark_accs = []
        for c in self.selected_clients:
            w_acc = test_watermark(c.model,c.key['x'],c.key['b'],self.device)
            watermark_accs.append(w_acc)
        print("Averaged Watermark Accurancy: {:.4f}".format(np.mean(watermark_accs)))
        # return np.mean(watermark_accs)
    
    # 所有客户端的水印相互测试
    def watermark_for_allclients(self):
        all_client_acc = []
        for ckey in self.clients:
            one_key_acc = []
            for cmodel in self.clients:
                w_acc = test_watermark(cmodel.model,ckey.key['x'],ckey.key['b'],self.device)
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

    def set_clients(self, clientObj):
        public_key = get_key(self.model,self.args.watermark_bits,self.args.use_watermark,"Conv2d")
        for i, train_slow, send_slow, malignant in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients,self.malignant_clients):
            train_data = read_client_data(self.dataset, self.datadir, i, is_train=True)
            test_data = read_client_data(self.dataset, self.datadir, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            malignant=malignant
                            )
            client.public_key = public_key
            self.clients.append(client)

    def set_malignant_clients(self):
        self.malignant_clients = self.select_slow_clients(
            self.malicious_frac
        )        

