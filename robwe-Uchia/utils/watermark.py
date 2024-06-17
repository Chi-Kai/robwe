import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_X(net_glob, embed_dim, use_watermark=False, layer_type="rep"):
    if layer_type == "rep":
        p = net_glob.rep_params()
    elif layer_type == "head":
        p = net_glob.head_params()
    elif layer_type == "bn":
        p = net_glob.bn_params()
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


def get_key(net_glob, embed_dim, use_watermark=False, layer_type="rep"):
    key = {}
    if not use_watermark:
        key = None
        return key
    key["x"] = get_X(net_glob, embed_dim, use_watermark, layer_type)
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
def get_layer_weights_and_predict(model, x, x_l, x_i, device, layer_type="rep"):
    if isinstance(model, nn.Module):
        # p = model.head_params()
        # p = p.cpu().view(1, -1).detach().numpy()
        if layer_type == "rep":
            p = model.rep_params()
            y = torch.tensor([], dtype=torch.float32).to(device)
            for i in p:
                y = torch.cat((y, i.view(-1)), 0)
            y = y.view(1, -1).to(device)
            start = x_i * x_l
            end = start + x_l
            y = y[:, start:end]
        elif layer_type == "head":
            p = model.head_params()
            y = torch.tensor([], dtype=torch.float32).to(device)
            for i in p:
                y = torch.cat((y, i.view(-1)), 0)
            y = y.view(1, -1).to(device)
        elif layer_type == "bn":
            p = model.bn_params()
            y = torch.tensor([], dtype=torch.float32).to(device)
            for i in p:
                y = torch.cat((y, i.view(-1)), 0)
            y = y.view(1, -1).to(device)

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


def test_watermark(model, x, b, x_l, x_i, device, layer_type="rep"):
    pred_b = get_layer_weights_and_predict(model, x, x_l, x_i, device, layer_type)
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
    p, keys, b_l = get_watermark(n_parties, m)
    keys, x = get_partition(net, b_l, n_parties, keys)
    return p, keys, x, b_l


def watermark_attack(key, frac, f=0):
    if f == 0:
        key = watermark_forgery(key, frac)
    return key


# 水印loss
class Signloss:
    def __init__(self, key, model, scheme, device):
        super(Signloss, self).__init__()
        self.alpha = 0.2  # self.sl_ratio
        self.loss = 0
        self.model = model
        self.key = key
        self.scheme = scheme  # scheme 为 水印嵌入的方式
        if key != None:
            self.x = key["x"]
            self.b = key["b"]
        self.device = device

    def get_loss(self, layer_type="rep"):
        self.reset()
        if self.key == None:
            return 0
        if self.scheme == 0:  # 方式0 水印嵌入在head层几个fc参数拼接的矩阵中
            if layer_type == "rep":
                p = self.model.rep_params()
            elif layer_type == "head":
                p = self.model.head_params()
            elif layer_type == "bn":
                p = self.model.bn_params()
            weights = torch.tensor([], dtype=torch.float32).to(self.device)
            for i in p:
                weights = torch.cat((weights, i.view(-1)), 0)
            # weights = weights.view(1, -1).to(self.device)
            loss = self.alpha * torch.sum(
                F.binary_cross_entropy(
                    input=torch.sigmoid(torch.matmul(weights, self.x.to(self.device))),
                    target=self.b.to(self.device),
                )
            )
        elif self.scheme == 1:  # 方式1 根据每个层的参数比例进行嵌入
            pass
        return loss

    def reset(self):
        self.loss = 0
