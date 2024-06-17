import copy
import math
import random
import torch
import torch.nn as nn


def get_X(net_glob, embed_dim, use_watermark=False):
    p = net_glob.rep_params()
    X_rows = 0
    for i in p:
        X_rows += i.numel()
    X_cols = embed_dim
    if not use_watermark:
        X = None
        return X
    X = torch.randn(X_rows, X_cols)
    return X


def get_b(embed_dim, use_watermark=False):
    if not use_watermark:
        b = None
        return b
    b = torch.sign(torch.rand(embed_dim) - 0.5)
    return b


def get_key(net_glob, embed_dim, use_watermark=False):
    key = {}
    if not use_watermark:
        key = None
        return key
    key["x"] = get_X(net_glob, embed_dim, use_watermark)
    key["b"] = get_b(embed_dim, use_watermark)
    return key


def watermark_forgery(key, frac):
    key_copy = copy.deepcopy(key)
    b = key_copy["b"].clone()
    s = frac * len(b)
    l = random.sample(range(len(b)), int(s))
    for i in l:
        b[i] = -b[i]
    key_copy["b"] = b

    mask = torch.zeros_like(key_copy["x"])
    num_replace = int(frac * key_copy["x"].numel())
    replace_idx = torch.randperm(key_copy["x"].numel())[:num_replace]
    mask.view(-1)[replace_idx] = 1
    replace_vals = torch.randn_like(key_copy["x"])
    key_copy["x"] = key_copy["x"] * (1 - mask) + replace_vals * mask

    return key_copy


#
def get_layer_weights_and_predict(model, x, x_l, x_i, device):
    if isinstance(model, nn.Module):
        # p = model.head_params()
        # p = p.cpu().view(1, -1).detach().numpy()
        p = model.rep_params()
        y = torch.tensor([], dtype=torch.float32).to(device)
        for i in p:
            y = torch.cat((y, i.view(-1)), 0)
        y = y.view(1, -1).to(device)
        start = x_i * x_l
        end = start + x_l
        y = y[:, start:end]

    elif isinstance(model, dict):
        raise Exception("model is a dict")
    pred_bparam = torch.matmul(y, x.to(device))  # dot product np.dot是矩阵乘法运算
    # print(pred_bparam)
    pred_bparam = torch.sign(pred_bparam.to(device))
    # pred_bparam = (pred_bparam + 1) / 2
    return pred_bparam


def compute_BER(pred_b, b, device):
    # correct_bit = torch.logical_not(torch.logical_xor(pred_b, b.to(device)))
    # correct_bit_num = torch.sum(correct_bit)
    # print(correct_bit_num,pred_b.size(),pred_b.size(0))
    correct_bit_num = torch.sum(pred_b == b.to(device))
    res = correct_bit_num / pred_b.size(1)
    return res.item()


def test_watermark(model, x, b, x_l, x_i, device):
    pred_b = get_layer_weights_and_predict(model, x, x_l, x_i, device)
    # print('pred_b',pred_b)
    # print('b',b)
    res = compute_BER(pred_b, b, device)
    return res


def get_partition(net, b_l, n_parties, keys):
    p = net.rep_params()
    X_rows = 0
    for i in p:
        X_rows += i.numel()
    x = int(X_rows / n_parties)
    for i in range(n_parties):
        keys[i]["x"] = torch.randn(x, b_l)
        keys[i]["id"] = i
    return keys, x


def get_watermark(n_parties, m):
    k = math.ceil(math.log2(n_parties))
    p_w = torch.sign(torch.rand(m) - 0.5)
    keys = {}

    binary_list = [format(i, "0{}b".format(k)) for i in range(2**k)]
    vectors = []
    for binary in binary_list:
        vector = torch.tensor([int(bit) * 2 - 1 for bit in binary])
        vectors.append(vector)
    t = torch.stack(vectors[:n_parties])

    for i in range(n_parties):
        keys[i] = {}
        keys[i]["b"] = torch.cat((p_w, t[i]), 0)
    p_w_repeat = p_w.repeat(t.shape[0], 1)
    p = torch.cat((p_w_repeat, t), 1)
    p = p.view(1, -1)
    return p, keys, k + m


def get_keys(net, n_parties, m):
    # p, keys, b_l = get_watermark(n_parties, m)
    keys = [{} for _ in range(n_parties)]
    for i in range(n_parties):
        keys[i]["b"] = get_b(m, True)
    keys, x = get_partition(net,m, n_parties, keys)
    return None, keys, x, m


def watermark_attack(key, frac, f=0):
    if f == 0:
        key = watermark_forgery(key, frac)
    return key
