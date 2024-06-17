#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: n")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=11, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--alg', type=str, default='fedrep', help='FL algorithm to use')
    # This program implements FedRep under the specification
    # --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg),
    # FedAvg (--alg fedavg) and FedProx (--alg prox)

    # algorithm-specific hyperparameters
    parser.add_argument('--local_rep_ep', type=int, default=1,
                        help="the number of local epochs for the representation for FedRep")
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    #parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--partition', type=str, default='noniid-#label4', help='the data partitioning strategy')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')
    parser.add_argument('--datadir', type=str, required=False, default="./data", help="Data directory")
    parser.add_argument('--save_path', type=str, required=False, default="./save", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')

    # build watermark
    parser.add_argument('--use_watermark', type=bool, default=True, help="决定是否要用水印")
    parser.add_argument('--use_rep_watermark', type=bool, default=True, help="决定是否要用表示层水印")
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入的水印有多长')   #私有层水印长度
    parser.add_argument('--rep_bit', type=int, default=100, help='rep bit')         #表示层（公共层），全部总的长度
    # parser.add_argument('--front_size', type=int, default=100, help='front m bit')   #公共层，感觉无用
    parser.add_argument('--malicious_frac', type=float, default=0.1,
                        help="the fraction of malicious clients")
    parser.add_argument('--tampered_frac', type=float, default=0.1,
                        help="proportion of watermarks tampered by malicious nodes")
    parser.add_argument('--detection', type=bool, default=True, help="决定检测恶意客户端")
    parser.add_argument('--scale', type=float, default=0.1, help="regularized loss前面的系数有多大")


    
    parser.add_argument('--confidence_level_nor', type=float, default=0.997,
                        help="confidence_level_nor")
    parser.add_argument('--confidence_level_bad', type=float, default=0.5,
                        help="confidence_level_bad")
    parser.add_argument('--bad_nums', type=int, default=7, help="bad_nums")

    args = parser.parse_args()
    return args
