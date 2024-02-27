import copy
import csv
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# 生成水印
def get_X(net_glob, embed_dim, use_watermark=False, layer_type="Linear"):
    p = net_glob.head if layer_type == "Linear" else net_glob.base.children()  # 使用layer_type选择不同层
    X_rows = 0
    for i in p:
        if layer_type == "Linear" and isinstance(i, nn.Linear):
            X_rows += i.weight.numel()
        elif layer_type == "Conv2d" and isinstance(i, nn.Conv2d):
            X_rows += i.weight.numel()
    X_cols = embed_dim
    if not use_watermark:
        X = None
        return X
    X = torch.randn(X_rows, X_cols)
    return X

def get_b(embed_dim,use_watermark=False):
    if not use_watermark:
        b = None
        return b
    b = torch.sign(torch.rand(embed_dim) - 0.5)
    return b

def get_key(net_glob, embed_dim, use_watermark=False, layer_type="Linear"):
    key = {}
    if not use_watermark:
        key = None
        return key
    key['x'] = get_X(net_glob, embed_dim, use_watermark, layer_type)
    key['b'] = get_b(embed_dim, use_watermark)
    return key

# 水印检测
def get_layer_weights_and_predict(model,x,device,layer_type="Linear"):
    # 得到参数 转化为一维向量
    if layer_type == "Linear":    
        weights = get_fc_weights(model)
    elif layer_type == "Conv2d":
        weights = get_conv_weights(model)
    pred_bparam = torch.matmul(weights, x.to(device))  # dot product np.dot是矩阵乘法运算
    pred_bparam = torch.sign(pred_bparam.to(device))
    return pred_bparam

# 计算正确的比特数
def compute_BER(pred_b, b, device):
    correct_bit_num = torch.sum(pred_b == b.to(device))
    res = correct_bit_num / pred_b.size(0)
    return res.item()

def test_watermark(model,x,b,device,layer_type="Linear"):
    pred_b = get_layer_weights_and_predict(model,x,device,layer_type)
    # print(pred_b.size())
    res = compute_BER(pred_b, b,device)
    return res

# 将总的嵌入区域分成若干个小区域,每个client只能修改一个小区域
# 传入的矩阵是二维的，根据参与方的数量，将矩阵分成若干个小矩阵
# 每个小矩阵的大小为总矩阵的1/n
# 参数: X:总的嵌入区域 b_l:每个client的水印长度 n_parties:参与方的数量
# 返回值: keys:每个client的水印 x:每个client的嵌入区域大小
def get_partition(net,b_l,n_parties,keys):
    p = net.rep_params()
    X_rows = 0
    for i in p:
        X_rows += i.numel()
    # 将x_rows分成n_parties份,每份大小相等
    x = int(X_rows / n_parties)
    for i in range(n_parties):
        keys[i]['x'] = torch.randn(x, b_l)
        keys[i]['id'] = i
    return keys,x                 


# 分配给每个client的水印,返回总的公共水印和每个client的水印
# 参数: n_parties:参与方的数量 m:前缀水印的长度
# 返回值: p:总的公共水印 keys:每个client的水印 b_l:每个client的水印长度
def get_watermark(n_parties,m):
    # 根据参与方的数量，水印是紧密的，比如2个人，每个人的水印都是1位
    # 寻找能表示n_parties的最小二进制位数
    k = math.ceil(math.log2(n_parties))
    # 生成一个m位的随机数
    p_w = torch.sign(torch.rand(m) - 0.5)
    keys = {}

    # 生成二进制数
    binary_list = [format(i, '0{}b'.format(k)) for i in range(2**k)]
    # 将二进制数中的0替换为-1，生成向量
    vectors = []
    for binary in binary_list:
        vector = torch.tensor([int(bit)*2-1 for bit in binary])
        vectors.append(vector)
    # 将向量堆叠成张量
    t = torch.stack(vectors[:n_parties])
    # 生成[1,k]的矩阵，总共n_parties个，每个都不相同
    # t = torch.sign(torch.randn(n_parties, k)-0.5)
    #print(t)
    # 如果有重复的，重新生成
    #while t.shape[0] > 1 and torch.unique(t, dim=0).shape[0] != t.shape[0]:
    #    t = torch.sign(torch.randn(n_parties, k)-0.5)
    # 将p_w 作为前缀和t中的每一行拼接作为每个client的水印
    for i in range(n_parties):
        keys[i] = {}
        keys[i]['b'] = torch.cat((p_w, t[i]), 0)
    #每个client的水印合并成一个总的水印
    p_w_repeat = p_w.repeat(t.shape[0],1)
    p = torch.cat((p_w_repeat,t),1)
    p = p.view(1,-1)
    return p,keys,k+m

# 使用前面两个函数，得到每个client的key
def get_keys(net,n_parties,m):
    p,keys,b_l = get_watermark(n_parties,m)
    keys,x = get_partition(net,b_l,n_parties,keys)
    return p,keys,x,b_l

# 水印攻击
# 将 key 中的 b 选取比例为frac的比特位取反
def watermark_forgery(key, frac):
    key_copy = copy.deepcopy(key)
    b = key_copy['b'].clone()
    s = frac * len(b)
    l = random.sample(range(len(b)), int(s))
    for i in l:
        b[i] = -b[i]
    key_copy['b'] = b

    # 替换掉原来的x
    # mask = torch.zeros_like(key_copy['x'])
    # num_replace = int (frac * key_copy['x'].numel())
    # replace_idx = torch.randperm(key_copy['x'].numel())[:num_replace]
    # mask.view(-1)[replace_idx] = 1
    # replace_vals = torch.randn_like(key_copy['x'])
    # key_copy['x'] = key_copy['x'] * (1 - mask) + replace_vals * mask

    return key_copy

# 水印篡改攻击
# f = 0: 随机篡改一定比例的比特位
# f = 1: 篡改front的比特位
# f = 2: 篡改back的比特位
def watermark_attack(key,frac,f = 0):
    if f == 0:
        key = watermark_forgery(key, frac)
    return key

# 水印loss    
class Signloss():
    def __init__(self,key, model,scheme,device):
        super(Signloss, self).__init__()
        self.alpha = 0.2  #self.sl_ratio
        self.loss = 0
        self.model = model
        self.key = key
        self.scheme = scheme # scheme 为 水印嵌入的方式
        if key != None:
            self.x = key['x']
            self.b = key['b']
        self.device = device
    def get_loss(self,layer_type="Linear"):
        self.reset()
        if self.key == None:
            return 0
        if self.scheme == 0: # 方式0 水印嵌入在head层几个fc参数拼接的矩阵中
           if layer_type == "Linear":    
                weights = get_fc_weights(self.model)
           elif layer_type == "Conv2d":
                weights = get_conv_weights(self.model)
           loss = self.alpha * torch.sum(
                             F.binary_cross_entropy(
                                 input=torch.sigmoid(torch.matmul(weights,self.x.to(self.device))),
                                 target=self.b.to(self.device)
                                 ))
        elif self.scheme == 1: #方式1 根据每个层的参数比例进行嵌入
            pass
        return loss 
    def reset(self):
        self.loss = 0

# 公共水印loss    
class PublicSignloss():
    def __init__(self,key, model,scheme,device):
        self.alpha = 0.2  #self.sl_ratio
        self.loss = 0
        self.model = model
        self.key = key
        self.scheme = scheme # scheme 为 水印嵌入的方式
        if key != None:
            self.x = key['x']
            self.b = key['b']
        self.device = device
    def get_loss(self):
        self.reset()
        if self.key == None:
            return 0
        if self.scheme == 0: # 方式0 水印嵌入在head层几个fc参数拼接的矩阵中
           weights = get_conv_weights(self.model)
           loss = self.alpha * torch.sum(
                             F.binary_cross_entropy(
                                 input=torch.sigmoid(torch.matmul(weights,self.x.to(self.device))),
                                 target=self.b.to(self.device)
                                 ))
        elif self.scheme == 1: #方式1 根据每个层的参数比例进行嵌入
            pass
        return loss 
    def reset(self):
        self.loss = 0   
    
def get_fc_weights(model):
    weights = []
    for fc_layer in model.head:
        # 获取全连接层的权重参数
      if isinstance(fc_layer,nn.Linear):
        weight = fc_layer.weight.view(-1)  # 展平为一维向量
        weights.append(weight)

    # 将权重参数拼接为一个一维向量
    concatenated_weights = torch.cat(weights)
    return concatenated_weights

def get_conv_weights(model):
    weights = []
    for conv_layer in model.base.children():
        if isinstance(conv_layer,nn.Conv2d):
            weight = conv_layer.weight.view(-1)
            weights.append(weight)

    # 将权重参数拼接为一个一维向量
    concatenated_weights = torch.cat(weights)
    return concatenated_weights

# 文件保存    
def tocsv(file, data):
    # 如果目录不存在
    directory = os.path.dirname(file)  # 获取文件所在目录
    if not os.path.exists(directory):  # 检查目录是否存在
        os.makedirs(directory)  # 创建目录
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if isinstance(data, dict):
            # 写入列名
            writer.writerow(data.keys())
            # 写入数据
            rows = zip(*[v if isinstance(v, (list, tuple, range)) 
                         else [v] for v in data.values()])
            writer.writerows(rows)
        elif isinstance(data, list):
            # 写入数据
            writer.writerows(data)