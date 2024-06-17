import torch
import torch.nn as nn
import numpy as np
import time
from utils.watermark import get_key,Signloss
from flcore.clients.clientbase import Client
#from utils.privacy import *
from utils.data_utils import read_client_data, DatasetSplit
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.preprocessing import label_binarize

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.is_watermark = args.use_watermark and args.watermark_bits > 0
        self.layer_type = args.layer_type
        if self.is_watermark:
            self.key = get_key(self.model,args.watermark_bits,args.use_watermark,layer_type=args.layer_type)

    def train(self,dataset,idxs):
        trainloader = self.load_train_data(dataset,idxs)
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        '''
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        '''
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                #水印
                if self.is_watermark:
                    loss += Signloss(self.key,self.model,0,self.device).get_loss(layer_type=self.layer_type)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        '''
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
        '''
    # 重载函数
    def load_train_data(self, dataset,idxs,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        #train_data = read_client_data(self.dataset, self.datadir, self.id, is_train=True)
        train_data = DatasetSplit(dataset,idxs)
        return DataLoader(train_data, batch_size,shuffle=True)

    def load_test_data(self, dataset,idxs,batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        #test_data = read_client_data(self.dataset, self.datadir, self.id, is_train=False)
        test_data = DatasetSplit(dataset,idxs)
        return DataLoader(test_data, batch_size, shuffle=True)

    def test_metrics(self,dataset,idxs):
        testloaderfull = self.load_test_data(dataset,idxs)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self,dataset,idxs):
        trainloader = self.load_train_data(dataset,idxs)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num