import copy
import itertools
import json
import numpy as np
import pandas as pd
import torch
import os
from scipy.stats import norm
#from utils.trainer_private import TesterPrivate
from utils.utils import construct_passport_kwargs

from utils.options import args_parser
from utils.train_utils import get_model, getdata
from models.Update import LocalUpdate
from models.test import test_img_local_all
from utils.watermark import (
    get_key,
    watermark_forgery,
    test_watermark,
    get_keys,
    watermark_attack,
)

import time
import random


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)


def generate_signature_dict(args, model):
    l = []
    for i in range(args.num_users):
        if i < args.malicious_frac * args.num_users:
            l.append(1)
        elif i < args.num_users:
            l.append(2)
        else:
            l.append(0)
    np.random.shuffle(l)
    keys = []
    key1 = get_key(
        net_glob=model, embed_dim=args.rep_bit, use_watermark=args.use_watermark
    )
    key2 = watermark_forgery(key1, args.tampered_frac)
    # print(key1['b'])
    # print(key2['b'])
    print((key1["b"] != key2["b"]).sum().item())
    for i in range(args.num_users):
        if l[i] == 1:
            keys.append(key2)
        if l[i] == 2:
            keys.append(key1)
        if l[i] == 0:
            keys.append(None)
    return keys


def get_dict(args, model):
    l = []
    for i in range(args.num_users):
        if i < args.malicious_frac * args.num_users:
            l.append(1)
        elif i < args.num_users:
            l.append(2)
        else:
            l.append(0)
    np.random.shuffle(l)
    p_w, key1, x_l, water_l = get_keys(
        net=model, n_parties=args.num_users, m=args.front_size
    )
    key2 = copy.deepcopy(key1)
    malice_client = []
    for i in range(args.num_users):
        if l[i] == 1:
            key2[i] = watermark_attack(key2[i], args.tampered_frac)
            xor_b = (key1[i]["b"] != key2[i]["b"]).sum().item()
            xor_x = (key1[i]["x"] != key2[i]["x"]).sum().item()
            print(
                "client {} is malicious,tampered_frac {} and {}".format(i, xor_b, xor_x)
            )
            malice_client.append(i)
    save_path = args.save_path + "/malice/" + str(args.frac) + "/" + str(args.epochs)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = (
        save_path
        + "/"
        + str(args.malicious_frac)
        + "_"
        + str(args.tampered_frac)
        + "_"
        + "malice_clients.txt"
    )
    with open(file_path, "w") as f:
        f.write(str(malice_client))
    return key1, key2, p_w, water_l, x_l, malice_client


def calculate_normal_distribution(a):
    mean = np.mean(a)
    std = np.std(a)
    dist = norm(loc=mean, scale=std)
    return dist, std, mean


def check_confidence_interval(
    nor, bad, b, confidence_level_nor, confidence_level_bad, bad_nums
):
    dist_nor, std_nor, mean_nor = calculate_normal_distribution(nor)
    if len(bad) >= bad_nums:
        dist_bad, std_bad, mean_bad = calculate_normal_distribution(bad)
        lower_bad, upper_bad = dist_bad.interval(confidence_level_bad)
        if std_nor == 0:
            return b >= upper_bad
        lower_nor, upper_nor = dist_nor.interval(confidence_level_nor)
        return b >= upper_bad
    else:
        lower_nor, upper_nor = dist_nor.interval(confidence_level_nor)
        return b >= lower_nor


def main(args, seed):
    init_seed(seed=seed)
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )
    lens = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = getdata(args)
    save_path = (
        args.save_path
        + "/dataset/"
        + str(args.frac)
        + "_"
        + str(args.embed_dim)
        + "_"
        + str(args.epochs)
    )
    if not os.path.exists(args.save_path + "/dataset/"):
        os.makedirs(args.save_path + "/dataset/")
    torch.save(dict_users_train, save_path + "_dataset_train.pt")
    torch.save(dict_users_test, save_path + "_dataset_test.pt")

    args.weight_type = "gamma"
    root = "pflipr-master/robwe/"
    if args.model == "resnet":
        args.passport_config = json.load(open(root + "configs/resnet18_passport.json"))
    if args.model == "alexnet":
        args.passport_config = json.load(open(root + "configs/alexnet_passport.json"))
    if args.model == "cnn":
        args.passport_config = json.load(open(root + "configs/cnn_passport.json"))

    model_glob = get_model(args).to(args.device)
    model_glob.train()

    if args.load_fed != "n":
        fed_model_path = args.load_fed
        model_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(model_glob.state_dict().keys())
    # print('net_glob.state_dict().keys():')
    # print(model_glob.state_dict().keys())
    model_keys = [*model_glob.state_dict().keys()]

    # build watermark----------------------------------------------------------------------------------------------
    # dict_X = get_X(net_glob=model_glob, embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    # dict_b = get_b(embed_dim=args.embed_dim, num_users=args.num_users,use_watermark=args.use_watermark)
    # build watermark----------------------------------------------------------------------------------------------

    keys = []
    keys_server, keys_rep, public_w, water_l, x_l, server_malice_clients = get_dict(
        args, model_glob
    )
    print(
        "watermark length:{},public_water: {},Matrix length: {}".format(
            water_l, public_w, x_l
        )
    )

    for i in range(args.num_users):
        key = get_key(model_glob, args.embed_dim, args.use_watermark,layer_type=args.layer_type)
        keys.append(key)

    save_path = (
        args.save_path
        + "/watermark/"
        + str(args.frac)
        + "/"
        + str(args.embed_dim)
        + "/"
        + str(args.epochs)
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    """
    torch.save(dict_X, save_path + '/dict_X.pt')
    torch.save(dict_b, save_path + '/dict_b.pt')
    """
    torch.save(keys, save_path + "/keys.pt")
    torch.save(keys_rep, save_path + "/keys_rep.pt")

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    layer_base = []
    if args.alg == "fedrep" or args.alg == "fedper":
        if args.model == "cnn":
            layer_base = [model_glob.weight_keys[i] for i in [0, 1, 2]]
        elif args.model == "alexnet":
            layer_base = [model_glob.weight_keys[i] for i in [0, 1, 5]]
    if args.alg == "fedavg":
        layer_base = []
    layer_base = list(itertools.chain.from_iterable(layer_base))

    # generate list of local models for each user
    model_clients = {}
    for user in range(args.num_users):
        model_local_dict = {}
        for key in model_glob.state_dict().keys():
            model_local_dict[key] = model_glob.state_dict()[key]
        model_clients[user] = model_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    success_rates = []
    start = time.time()
    all_one_for_all_clients_rates = []
    all_epochs_malice_clients = []
    all_epochs_client_water_accs = []
    malice_client_detect = []
    malice_client_detect_rates = []
    false_detect = []
    false_detect_rates = []
    all_detect_malice_clients = []
    all_detect_malice_clients_rates = []
    rep_accs = []
    server_accs = []
    client_sample_times = [0] * args.num_users
    server_water_accs = [0] * args.num_users
    client_waters = [0] * args.num_users
    for iter in range(args.epochs):
        state_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # if iter == args.epochs:
        #    m = args.num_users

        idxs_users = np.random.choice(
            [
                user
                for user in range(args.num_users)
                if user not in malice_client_detect
                and user not in false_detect
                and user != 99
            ],
            m,
            replace=False,
        )
        idxs_users.sort()

        total_len = 0
        last = iter == args.epochs

        server_memory = {}
        malice_client = []
        client_accs_for_iter = {}
        if iter == 0:
            client_water_accs = [0] * args.num_users
        else:
            client_water_accs = copy.deepcopy(all_epochs_client_water_accs[-1])

        for ind, idx in enumerate(idxs_users):
            pre_model = copy.deepcopy(model_glob)
            pre_model.load_state_dict(model_clients[idx])
            server_water_accs[idx] = test_watermark(
                pre_model,
                keys_server[idx]["x"],
                keys_server[idx]["b"],
                x_l,
                keys_server[idx]["id"],
                args.device,
            )
        for idx in idxs_users:
            client_sample_times[idx] += 1

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if args.epochs == iter:
                client = LocalUpdate(
                    args=args,
                    dataset=dataset_train,
                    idxs=dict_users_train[idx][: args.m_ft],
                    key=keys[idx],
                    rep_x=keys_rep[idx]["x"],
                    rep_b=keys_rep[idx]["b"],
                    x_l=x_l,
                    x_i=keys_rep[idx]["id"],
                )
            else:
                client = LocalUpdate(
                    args=args,
                    dataset=dataset_train,
                    idxs=dict_users_train[idx][: args.m_tr],
                    key=keys[idx],
                    rep_x=keys_rep[idx]["x"],
                    rep_b=keys_rep[idx]["b"],
                    x_l=x_l,
                    x_i=keys_rep[idx]["id"],
                )

            model_client = copy.deepcopy(model_glob)
            state_client = model_client.state_dict()
            if args.alg != "fedavg":
                for k in model_clients[idx].keys():
                    if k not in layer_base:
                        state_client[k] = model_clients[idx][k]
            model_client.load_state_dict(state_client)

            print("client {} start training".format(idx))
            state_client, loss, indd, net = client.train(
                net=model_client.to(args.device),
                w_glob_keys=layer_base,
                lr=args.lr,
                last=last,
                args=args,
            )
            if args.use_watermark:
                test_model = copy.deepcopy(model_glob)
                test_model.load_state_dict(state_client)
                success_rate = test_watermark(
                    test_model,
                    keys[idx]["x"],
                    keys[idx]["b"],
                    0,
                    0,
                    args.device,
                    args.layer_type
                )
                # success_rate = client.validate(net=model_client.to(args.device),device=args.device)
                success_rates.append(success_rate)
                # print(success_rate)

            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            server_memory[idx] = copy.deepcopy(state_client)
            save_head(args=args, idx=idx, model=net)

        print("epoch {} start test".format(iter))
        # print('server memory length: {},keys: {}'.format(len(server_memory),server_memory.keys()))
        for ind, i in enumerate(idxs_users):
            client_model = copy.deepcopy(model_glob)
            client_model.load_state_dict(server_memory[i])
            client_acc = test_watermark(
                client_model,
                keys_server[i]["x"],
                keys_server[i]["b"],
                x_l,
                keys_server[i]["id"],
                args.device,
            )
            client_acc_2 = test_watermark(
                client_model,
                keys_rep[i]["x"],
                keys_rep[i]["b"],
                x_l,
                keys_rep[i]["id"],
                args.device,
            )
            client_waters[i] = client_acc
            client_accs_for_iter[i] = client_acc
        for ind, i in enumerate(idxs_users):
            # other_client_accs = []
            # other_client_test_accs = []
            other_client_accs_in_i = []
            malice_client_accs = []
            same_epochs_clients = [
                ind
                for ind, j in enumerate(client_sample_times)
                if j == client_sample_times[i]
                and ind not in malice_client_detect
                and ind not in false_detect
            ]
            print("---------------------------------")
            print(
                "client {} start test,xor_frac: {}, server_acc: {}".format(
                    i,
                    (keys_server[i]["b"] != keys_rep[i]["b"]).sum().item(),
                    client_accs_for_iter[i],
                )
            )
            print(
                "choose_nums: {} same_epochs_clients:{}".format(
                    client_sample_times[i], same_epochs_clients
                )
            )
            for ind, j in enumerate(same_epochs_clients):
                if i != j:
                    # other_client_acc = test_watermark(other_client_model,keys_server[i]['x'],keys_server[i]['b'],x_l,keys_server[i]['id'],args.device)
                    # other_client_test_acc = test_watermark(other_client_model,keys_server[j]['x'],keys_server[j]['b'],x_l,keys_server[j]['id'],args.device)
                    # other_client_acc_in_i = test_watermark(client_model,keys_server[j]['x'],keys_server[j]['b'],x_l,keys_server[j]['id'],args.device)
                    # other_client_accs.append(other_client_acc)
                    # other_client_test_accs.append(other_client_test_acc)
                    # if j in server_malice_clients:
                    #      malice_client_accs.append(client_waters[j])
                    # else:
                    other_client_accs_in_i.append(client_waters[j])
            # print('other_client_accs_in {} area:{}'.format(i,other_client_accs))
            print("other_area_accs_in_{}_client:{}".format(i, other_client_accs_in_i))
            # print('malice_client_accs:{}'.format(malice_client_accs))
            if (
                len(other_client_accs_in_i) == 0
                or len(other_client_accs_in_i) < len(idxs_users) - 1
            ):
                print(
                    "client {} is advence,epochs is {}".format(
                        i, client_sample_times[i]
                    )
                )
                continue
            # avg_accs = sum(other_client_accs_in_i)/len(other_client_accs_in_i)
            # print('avg_accs:{}'.format(avg_accs))
            # dis = calculate_normal_distribution(other_client_accs_in_i)

            # is_in_interval = check_confidence_interval(other_client_accs_in_i,all_detect_malice_clients_rates,client_accs_for_iter[i], 0.997,0.5,7)

            is_in_interval = True
            if is_in_interval:
                client_water_accs[i] = client_accs_for_iter[i]
                continue
            else:
                print(
                    "client {} is malicious,water acc: {} is_in_interval: {} ".format(
                        i, client_accs_for_iter[i], is_in_interval
                    )
                )
                malice_client.append(keys_rep[i]["id"])
                if (
                    keys_rep[i]["id"] not in malice_client_detect
                    and keys_rep[i]["id"] in server_malice_clients
                ):
                    malice_client_detect.append(keys_rep[i]["id"])
                if (
                    keys_rep[i]["id"] not in false_detect
                    and keys_rep[i]["id"] not in server_malice_clients
                ):
                    false_detect.append(keys_rep[i]["id"])
                # keys_rep[i] = copy.deepcopy(keys_server[i])
                all_detect_malice_clients.append(keys_rep[i]["id"])
                all_detect_malice_clients_rates.append(client_accs_for_iter[i])
                print(
                    "all_detect_malice_clients_rates:{}".format(
                        all_detect_malice_clients_rates
                    )
                )
                server_memory[i] = None
                if iter == 0:
                    client_water_accs[i] = 0
                else:
                    client_water_accs[i] = copy.deepcopy(
                        all_epochs_client_water_accs[-1][i]
                    )
        if len(malice_client) == 0:
            malice_client_detect_rate = 0
        else:
            malice_client_detect_rate = len(malice_client_detect) / len(
                server_malice_clients
            )
        false_detect_rate = len(false_detect) / args.num_users

        print("malice_client:{}".format(malice_client))
        print("client_water_accs:{}".format(client_water_accs))
        print("detect rate: {}".format(malice_client_detect_rate))
        print("false detect rate: {}".format(false_detect_rate))

        all_epochs_malice_clients.append(malice_client)
        all_epochs_client_water_accs.append(client_water_accs)
        malice_client_detect_rates.append(malice_client_detect_rate)
        false_detect_rates.append(false_detect_rate)
        for ind, i in enumerate(idxs_users):
            if server_memory[i] is not None:
                if len(state_glob) == 0:
                    state_glob = copy.deepcopy(server_memory[i])
                    for k, key in enumerate(model_glob.state_dict().keys()):
                        state_glob[key] = state_glob[key] * lens[i]
                        model_clients[i][key] = server_memory[i][key]
                else:
                    for k, key in enumerate(model_glob.state_dict().keys()):
                        state_glob[key] += server_memory[i][key] * lens[i]
                        model_clients[i][key] = server_memory[i][key]
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in model_glob.state_dict().keys():
            state_glob[k] = torch.div(state_glob[k], total_len)

        state_client = model_glob.state_dict()
        for k in state_glob.keys():
            state_client[k] = state_glob[k]
        if args.epochs != iter:
            model_glob.load_state_dict(state_glob)

        attack_accs = []
        defend_accs = []
        for i in range(args.num_users):
            attack_acc = test_watermark(
                model_glob,
                keys_rep[i]["x"],
                keys_rep[i]["b"],
                x_l,
                keys_rep[i]["id"],
                args.device,
            )
            attack_accs.append(attack_acc)
            if keys_server[i]["id"] in all_detect_malice_clients:
                continue
            else:
                defend_acc = test_watermark(
                    model_glob,
                    keys_server[i]["x"],
                    keys_server[i]["b"],
                    x_l,
                    keys_server[i]["id"],
                    args.device,
                )
                defend_accs.append(defend_acc)
        rep_accs.append(sum(attack_accs) / len(attack_accs))
        server_accs.append(sum(defend_accs) / len(defend_accs))
        print(
            "round {} attacker_acc: {}, defender_acc: {}".format(
                iter,
                sum(attack_accs) / len(attack_accs),
                sum(defend_accs) / len(defend_accs),
            )
        )
        if iter == args.epochs - 1:
            print("defend accs:{}".format(defend_accs))
            print("attack accs:{}".format(attack_accs))
        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            acc_test, loss_test = test_img_local_all(
                model_glob,
                args,
                dataset_test,
                dict_users_test,
                w_glob_keys=layer_base,
                w_locals=model_clients,
                indd=indd,
                dataset_train=dataset_train,
                dict_users_train=dict_users_train,
                return_all=False,
            )
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally
            # updated versions of the global model at the end of each round)
            if args.use_watermark:
                water_acc = sum(success_rates) / len(success_rates)
            else:
                water_acc = 0
            if iter != args.epochs:
                print(
                    "Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f},Water acc: {:.2f}".format(
                        iter, loss_avg, loss_test, acc_test, water_acc
                    )
                )
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model,
                # we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print(
                    "Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}, Water acc: {:.2f}".format(
                        loss_avg, loss_test, acc_test, water_acc
                    )
                )

        if iter % args.save_every == args.save_every - 1:
            path = (
                args.save_path
                + "/models/"
                + str(args.frac)
                + "/"
                + str(args.embed_dim)
                + "/"
                + str(args.epochs)
            )
            if not os.path.exists(path):
                os.makedirs(path)
            model_save_path = (
                path
                + "/accs_"
                + args.alg
                + "_"
                + args.dataset
                + "_"
                + str(args.num_users)
                + "_iter"
                + str(args.use_watermark)
                + str(iter + 1)
                + ".pt"
            )
            torch.save(model_glob.state_dict(), model_save_path)

    if args.use_watermark:
        for i in range(args.num_users):
            model = copy.deepcopy(model_glob)
            model.load_state_dict(model_clients[i])
            one_for_all_clients_rates = []
            for j in range(args.num_users):
                sign_acc = test_watermark(
                    model,
                    keys[j]["x"],
                    keys[j]["b"],
                    0,
                    0,
                    args.device,
                    args.layer_type
                )
                one_for_all_clients_rates.append(sign_acc)
            all_one_for_all_clients_rates.append(one_for_all_clients_rates)

    # print(end - start)
    # print(times)
    # print(accs)

    accs_dir = args.save_path + "/accs/" + str(args.frac) + "/" + str(args.epochs)
    accs_csv = (
        accs_dir
        + "/accs_"
        + args.dataset
        + "_"
        + str(args.num_users)
        + "_"
        + str(args.use_watermark)
        + "_"
        + str(args.embed_dim)
        + ".csv"
    )
    if not os.path.exists(accs_dir):
        os.makedirs(accs_dir)
    accs = np.array(accs)
    df = pd.DataFrame({"round": range(len(accs)), "acc": accs})
    df.to_csv(accs_csv, index=False)

    if args.use_watermark:
        path = (
            args.save_path
            + "/water_acc"
            + "/"
            + str(args.frac)
            + "/"
            + str(args.epochs)
        )
        if not os.path.exists(path):
            os.makedirs(path)
        all_detect_all_dir = (
            path
            + "/all_detect_all_rate_"
            + args.dataset
            + "_"
            + str(args.num_users)
            + "_"
            + str(args.use_watermark)
            + "_"
            + str(args.embed_dim)
            + "_"
            + str(args.frac)
            + ".xlsx"
        )
        all_one_for_all_clients_rates = np.array(all_one_for_all_clients_rates)
        all_one_for_all_clients_rates = np.transpose(all_one_for_all_clients_rates)
        df = pd.DataFrame(all_one_for_all_clients_rates)
        df.to_excel(all_detect_all_dir)

        server_file = (
            path
            + "/server_acc_"
            + args.dataset
            + "_"
            + str(args.num_users)
            + "_"
            + str(args.use_watermark)
            + "_"
            + str(args.embed_dim)
            + "_"
            + str(args.rep_bit)
            + "_"
            + str(args.frac)
            + "_"
            + str(args.malicious_frac)
            + "_"
            + str(args.tampered_frac)
            + ".csv"
        )
        server_accs = np.array(server_accs)
        # 将server_accs,rep_accs,malice_client_detect_rates,false_detect_rates保存到csv文件中
        df = pd.DataFrame(
            {
                "server_accs": server_accs,
                "rep_accs": rep_accs,
                "malice_client_detect_rates": malice_client_detect_rates,
                "false_detect_rates": false_detect_rates,
            }
        )
        df.to_csv(server_file, index=False)

    path = args.save_path + "/malice" + "/" + str(args.frac) + "/" + str(args.epochs)
    if not os.path.exists(path):
        os.makedirs(path)
    malice_file = (
        path
        + "/malice_client_"
        + args.dataset
        + "_"
        + str(args.num_users)
        + "_"
        + str(args.use_watermark)
        + "_"
        + str(args.embed_dim)
        + "_"
        + str(args.rep_bit)
        + "_"
        + str(args.frac)
        + "_"
        + str(args.malicious_frac)
        + "_"
        + str(args.tampered_frac)
        + ".csv"
    )
    all_epochs_malice_clients = np.array(all_epochs_malice_clients)
    df = pd.DataFrame(all_epochs_malice_clients)
    df.to_csv(malice_file, index=False)
    path = (
        args.save_path
        + "/client_water_acc"
        + "/"
        + str(args.frac)
        + "/"
        + str(args.epochs)
    )
    if not os.path.exists(path):
        os.makedirs(path)
    client_water_file = (
        path
        + "/client_water_acc_"
        + args.dataset
        + "_"
        + str(args.num_users)
        + "_"
        + str(args.use_watermark)
        + "_"
        + str(args.embed_dim)
        + "_"
        + str(args.rep_bit)
        + "_"
        + str(args.frac)
        + "_"
        + str(args.malicious_frac)
        + "_"
        + str(args.tampered_frac)
        + ".csv"
    )
    all_epochs_client_water_accs = np.array(all_epochs_client_water_accs)
    df = pd.DataFrame(all_epochs_client_water_accs)
    df.to_csv(client_water_file, index=False)


def save_head(args, idx, model):
    save_dir = (
        args.save_path
        + "/head/"
        + str(args.frac)
        + "/"
        + str(args.embed_dim)
        + "/"
        + str(args.epochs)
        + "/"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    head_file = save_dir + str(idx) + ".npy"
    head = model.get_params()
    np.save(head_file, head, allow_pickle=True)


if __name__ == "__main__":
    args = args_parser()
    if args.embed_dim == 0:
        args.use_watermark = False
    main(args=args, seed=args.seed)
