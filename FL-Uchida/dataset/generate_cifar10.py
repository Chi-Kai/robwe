import argparse
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
# num_clients = 20
# num_classes = 10
dir_path = "Cifar10/"


# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition,alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + partition + str(alpha) + "/train/"
    test_path = dir_path + partition + str(alpha) + "/test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, alpha):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, alpha=alpha)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition,alpha=alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dir', "--dir_path", type=str, default="Cifar10/")
    parser.add_argument('-p', "--partition", type=str, default="pat")
    parser.add_argument('-nclient', "--num_clients", type=int, default=100)
    parser.add_argument('-nclasses', "--num_classes", type=int, default=10)
    parser.add_argument('-a', "--alpha", type=float, default=0.1)
    parser.add_argument('-niid', "--niid", type=bool, default=True)
    parser.add_argument('-b', "--balance", type=bool, default=False)
    
    args = parser.parse_args()
    generate_cifar10(args.dir_path, args.num_clients, args.num_classes, args.niid, args.balance, args.partition,args.alpha)