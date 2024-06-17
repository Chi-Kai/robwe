# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from models.layers.passportconv2d_private import PassportPrivateBlock

class LinearPrivateBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()

        self.linear = nn.Linear(i, o, bias=False)
        self.weight = self.linear.weight

        self.init_scale(True)
        self.init_bias(True)

        self.reset_parameters()
    
    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.linear.out_features).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None
    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.linear.out_features).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        pass

    def forward(self, x):
        x = self.linear(x)
        x = x * self.scale [None, :, None, None] + self.bias [None, :, None, None]
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = PassportPrivateBlock(1, 64, ks=5)
        self.conv2 = PassportPrivateBlock(64, 64, ks=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

        layer = [self.conv1,self.conv1,self.conv2,self.conv2]
        self.features = nn.Sequential(*layer)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def head_params(self):
        return [self.conv1.weight, self.conv2.weight]
    def get_params(self):
        state_dict = self.state_dict()
        params = {k: v.cpu().numpy() for k,v in state_dict.items() if 'fc' not in k}        
        return params

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv1 = PassportPrivateBlock(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv2 = PassportPrivateBlock(64, 64, 5)
        self.fc1 = nn.Linear(64 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

        layer = [self.conv1,self.pool,self.conv2,self.pool]
        self.features = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
    def head_params(self):
        return [self.conv1.weight, self.conv2.weight]
    def get_params(self):
        state_dict = self.state_dict()
        params = {k: v.cpu().numpy() for k,v in state_dict.items() if 'fc' not in k}        
        return params
    
class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.conv1 = PassportPrivateBlock(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = PassportPrivateBlock(64, 128, 5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,100)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

        layer = [self.conv1,self.pool,self.conv2,self.pool]
        self.features = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def head_params(self):
        return [self.conv1.weight, self.conv2.weight]
    def get_params(self):
        state_dict = self.state_dict()
        params = {k: v.cpu().numpy() for k,v in state_dict.items() if 'fc' not in k}        
        return params

class CNN_FEMNIST(nn.Module):
    def __init__(self, args):
        super(CNN_FEMNIST, self).__init__()
        self.conv1 = PassportPrivateBlock(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = PassportPrivateBlock(4, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
                            
        layer = [self.conv1,self.pool,self.conv2,self.pool]
        self.features = nn.Sequential(*layer)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def head_params(self):
        return [self.conv1.weight, self.conv2.weight]
    def get_params(self):
        state_dict = self.state_dict()
        params = {k: v.cpu().numpy() for k,v in state_dict.items() if 'fc' not in k}        
        return params

class RNNSent(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, args, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, emb_arr=None):
        super(RNNSent, self).__init__()
        VOCAB_DIR = 'models/embs.json'
        emb, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.encoder = torch.tensor(emb).to(args.device)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(nhid, 10)
        self.decoder = nn.Linear(10, ntoken)

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.device = args.device

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = torch.transpose(input, 0, 1)
        emb = torch.zeros((25, 4, 300))
        for i in range(25):
            for j in range(4):
                emb[i, j, :] = self.encoder[input[i, j], :]
        emb = emb.to(self.device)
        emb = emb.view(300, 4, 25)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(F.relu(self.fc(output)))
        decoded = self.decoder(output[-1, :, :])
        return decoded.t(), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class VGG16(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PassportPrivateBlock(128, 128, 3, 1),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            PassportPrivateBlock(128, 256, 3, 1),
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            PassportPrivateBlock(256, 256, 3, 1),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.weight_keys = [
            ["features[0].weight", "features[0].bias"],
            ["features[3].weight", "features[3].bias"],
            ["features[7].weight", "features[7].bias"],
            ["features[10].weight", "features[10].bias"],
            ["features[12].weight", "features[12].bias"],
            ["features[13].weight", "features[13].bias"],
            ["features[20].weight", "features[20].bias"],
            ["features[24].weight", "features[24].bias"],
            ["features[27].weight", "features[27].bias"],
            ["features[30].weight", "features[30].bias"],
            ["features[34].weight", "features[34].bias"],
            ["features[37].weight", "features[37].bias"],
            ["features[40].weight", "features[40].bias"],
            ["classifier[0].weight", "classifier[0].bias"],
            ["classifier[3].weight", "classifier[3].bias"],
            ["classifier[6].weight", "classifier[6].bias"],
        ]
        self.head_layers = [
            self.features[10].weight,
            self.features[12].weight,
            self.features[13].weight,
        ]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # 返回所有fc层权重,放到一个list里
    def head_params(self):
        return self.head_layers
    def get_params(self):
        return self.head_layers
    # 所有表示层参数
    def rep_params(self):
        return [
            self.classifier[0].weight,
            self.classifier[3].weight,
            self.classifier[6].weight,
        ]

