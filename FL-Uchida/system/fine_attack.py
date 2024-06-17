import copy
import argparse
import itertools
import time
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from utils.data_utils import getdata
from utils.watermark import test_watermark
from flcore.trainmodel.models import *
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverrep import FedRep


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


def test(model, device, dataloader):

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
            loss_meter += F.cross_entropy(
                pred, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = pred.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
            runcount += data.size(0)

    loss_meter /= runcount
    acc_meter /= runcount

    return loss_meter, acc_meter


def main(args, seed):
    init_seed(seed=seed)
    args.device = torch.device(
        "cuda:{}".format(args.device_id)
        if torch.cuda.is_available() and args.device_id != -1
        else "cpu"
    )
    dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)

    save_path = args.save_folder_name + "/dataset/" + str(args.watermark_bits)
    dict_users_train = torch.load(save_path + "_dataset_train.pt")
    dict_users_test = torch.load(save_path + "_dataset_test.pt")

    # model
    if args.dataset == "Cifar10":
        model = CNNCifar(num_classes=10).to(args.device)
    elif args.dataset == "Cifar100":
        model = CNNCifar100(num_classes=100).to(args.device)
    elif args.dataset == "Mnist":
        model = CNNMnist(num_classes=10).to(args.device)
    elif args.dataset == "Femnist":
        model = CNN_FEMNIST(num_classes=10).to(args.device)
        # select algorithm
    if args.algorithm == "FedAvg":
        head = copy.deepcopy(model.fc)
        model.fc = nn.Identity()
        model = BaseHeadSplit(model, head)
    model_path = (
        args.save_folder_name
        + "/models/"
        + args.dataset
        + "/"
        + "bits"
        + str(args.watermark_bits)
        + "_global.pt"
    )
    print(model_path)
    model.load_state_dict(torch.load(model_path))

    # key
    keys = [{} for i in range(args.num_clients)]
    for i in range(args.num_clients):
        keys[i] = torch.load(
            args.save_folder_name
            + "/keys/"
            + args.dataset
            + "/"
            + "bits"
            + str(args.watermark_bits)
            + "_"
            + str(i)
            + ".pt"
        )

    # init test
    model.eval()
    test_ldr_1 = DataLoader(
        DatasetSplit(dataset_test, dict_users_test[1]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_ldr_22 = DataLoader(
        DatasetSplit(dataset_test, dict_users_test[22]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    _, acc_1 = test(model, args.device, test_ldr_1)
    _, acc_22 = test(model, args.device, test_ldr_22)
    water_acc_1 = test_watermark(
        model, keys[1]["x"], keys[1]["b"], args.device, "Conv2d"
    )
    water_acc_22 = test_watermark(
        model, keys[22]["x"], keys[22]["b"], args.device, "Conv2d"
    )
    # all client water test
    all_water_acc = []
    for i in range(args.num_clients):
        all_water_acc.append(
            test_watermark(model, keys[i]["x"], keys[i]["b"], args.device, "Conv2d")
        )
    avg_water_acc = sum(all_water_acc) / len(all_water_acc)
    print(
        "Init test: acc_1:{},acc_22:{},water_acc_1:{},water_acc_22:{},avg_acc: {}".format(
            acc_1, acc_22, water_acc_1, water_acc_22, avg_water_acc
        )
    )

    # training

    train_ldr_1 = DataLoader(
        DatasetSplit(dataset_train, dict_users_train[1]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.local_learning_rate)
    water_acc = []
    accs_1 = []
    accs_22 = []
    avg_water_accs = []
    all_model_accs = []
    for iter in range(args.fine_epochs):
        model.train()
        loss_meter = 0
        acc_meter = 0
        runcount = 0
        for data, target in train_ldr_1:
            data, target = data.to(args.device), target.to(args.device)
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            loss_meter += loss.item()
            pred = output.max(1, keepdim=True)[1]
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
            runcount += data.size(0)
        loss = loss_meter / runcount
        acc_1 = acc_meter / runcount
        _, acc_22 = test(model, args.device, test_ldr_22)
        water_acc_1 = test_watermark(
            model, keys[1]["x"], keys[1]["b"], args.device, "Conv2d"
        )
        water_acc_22 = test_watermark(
            model, keys[22]["x"], keys[22]["b"], args.device, "Conv2d"
        )
        all_water_acc = []
        for i in range(args.num_clients):
            all_water_acc.append(
                test_watermark(model, keys[i]["x"], keys[i]["b"], args.device, "Conv2d")
            )
        avg_model_acc = 0
        all_model_accs.append(avg_model_acc)
        avg_water_acc = sum(all_water_acc) / len(all_water_acc)
        print(
            "Round {:3d}, loss {:.3f}, acc_1 {:.4f}, acc_22 {:.4f}, water_acc_1 {:.2f}, water_acc_22 {:.2f},avg_water_acc {:.2f},avg_model_acc {:.2f}".format(
                iter, loss, acc_1, acc_22, water_acc_1, water_acc_22, avg_water_acc,avg_model_acc
            )
        )
        water_acc.append(water_acc_22)
        avg_water_accs.append(avg_water_acc)
    # test model for all
    model.eval()
    test_ld = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    _,acc = test(model, args.device, test_ld)
    print("Final test acc: ",acc)
    all_model_accs[-1] = acc
    save_path = args.save_folder_name + "/fine_tune/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.DataFrame(
        {
            "water_acc": water_acc,
            "avg_water_acc": avg_water_acc,
            "avg_model_acc": avg_model_acc,
        }
    )
    df.to_csv(save_path + "/{}_{}acc.csv".format(args.watermark_bits, args.num_clients))


if __name__ == "__main__":

    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument(
        "-go", "--goal", type=str, default="test", help="The goal for this experiment"
    )
    parser.add_argument(
        "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-data", "--dataset", type=str, default="Cifar10")
    parser.add_argument("-nb", "--num_classes", type=int, default=10)
    parser.add_argument("-m", "--model", type=str, default="watercnn")
    parser.add_argument("-lbs", "--batch_size", type=int, default=10)
    parser.add_argument(
        "-lr",
        "--local_learning_rate",
        type=float,
        default=0.01,
        help="Local learning rate",
    )
    parser.add_argument("-ld", "--learning_rate_decay", type=bool, default=False)
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument("-gr", "--global_rounds", type=int, default=10)
    parser.add_argument(
        "-ls",
        "--local_epochs",
        type=int,
        default=11,
        help="Multiple update steps in one local epoch.",
    )
    parser.add_argument("-algo", "--algorithm", type=str, default="FedAvg")
    parser.add_argument(
        "-jr",
        "--join_ratio",
        type=float,
        default=1.0,
        help="Ratio of clients per round",
    )
    parser.add_argument(
        "-rjr",
        "--random_join_ratio",
        type=bool,
        default=False,
        help="Random ratio of clients per round",
    )
    parser.add_argument(
        "-nc", "--num_clients", type=int, default=2, help="Total number of clients"
    )
    parser.add_argument(
        "-pv", "--prev", type=int, default=0, help="Previous Running times"
    )
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument(
        "-eg", "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
    )
    parser.add_argument(
        "-dp", "--privacy", type=bool, default=False, help="differential privacy"
    )
    parser.add_argument("-dps", "--dp_sigma", type=float, default=0.0)
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items")
    parser.add_argument("-ab", "--auto_break", type=bool, default=False)
    parser.add_argument("-dlg", "--dlg_eval", type=bool, default=False)
    parser.add_argument("-dlgg", "--dlg_gap", type=int, default=100)
    parser.add_argument("-bnpc", "--batch_num_per_client", type=int, default=2)
    parser.add_argument("-nnc", "--num_new_clients", type=int, default=0)
    parser.add_argument("-fte", "--fine_tuning_epoch", type=int, default=0)
    parser.add_argument("-datadir", "--datadir", type=str, default="Cifar10/dir0.2/")
    # watermark
    parser.add_argument("-wbs", "--watermark_bits", type=int, default=50)
    parser.add_argument(
        "-w", "--use_watermark", type=bool, default=True, help="use watermark or no"
    )
    parser.add_argument(
        "-mfc",
        "--malicious_frac",
        type=float,
        default=0.0,
        help="The rate for malignant clients",
    )
    parser.add_argument(
        "-tfc",
        "--tampered_frac ",
        type=float,
        default=0.0,
        help="The rate for malignant clients",
    )
    parser.add_argument(
        "-bt", "--beta", type=float, default=0.5, help="beta for distribution"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="noniid-#label4",
        help="the data partitioning strategy",
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="Conv2d",
        help="the data partitioning strategy",
    )
    # practical
    parser.add_argument(
        "-cdr",
        "--client_drop_rate",
        type=float,
        default=0.0,
        help="Rate for clients that train but drop out",
    )
    parser.add_argument(
        "-tsr",
        "--train_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when training locally",
    )
    parser.add_argument(
        "-ssr",
        "--send_slow_rate",
        type=float,
        default=0.0,
        help="The rate for slow clients when sending global model",
    )
    parser.add_argument(
        "-ts",
        "--time_select",
        type=bool,
        default=False,
        help="Whether to group and select clients at each round according to time cost",
    )
    parser.add_argument(
        "-tth",
        "--time_threthold",
        type=float,
        default=10000,
        help="The threthold for droping slow clients",
    )
    # Ditto / FedRep
    parser.add_argument("-pls", "--plocal_steps", type=int, default=1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print(
            "Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma)
        )
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    print("datadir: {}".format(args.datadir))
    print("=" * 50)
    if args.use_watermark:
        print("watermark bits: {}".format(args.watermark_bits))
    args.fine_epochs = 25
    main(args=args, seed=1234)
