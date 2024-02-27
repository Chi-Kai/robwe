import csv
import os
import torch 
from models.alexnet import AlexNet

def construct_passport_kwargs(self):
    passport_settings = self.passport_config
    model = self.model_name 
    bit_length = self.num_bit
    dataset = self.dataset
    passport_kwargs = {}
    
    alexnet_channels = {
        '4': (384, 3456),
        '5': (256, 2304),
        '6': (256, 2304)
    }
    resnet_channels = {
        'layer2': (128, 1152),
        'layer3': (256, 2304),
        'layer4': (512, 4608)
    }

    cnn_channels = {
        '0': (64, 1600),
        '2': (128, 3200),
    }
    
    mnist_channels = {
        '0': (64, 1600),
        '2': (64, 1600),
    }

    for layer_key in passport_settings:
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:  # i = 0, 1 
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]: # module_key = convbnrelu
                    flag = passport_settings[layer_key][i][module_key] # flag = str in module_key
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True

                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag
                    }
                           
 #                  b = torch.sign(torch.rand(self.num_bit) - 0.5)
                    if b is not None:
                        
                        output_channels = int (bit_length * 512 / 2048)
                        output_channels = int (bit_length * 512 / 2048)
  
                        bsign = torch.sign(torch.rand(output_channels) - 0.5)
                        # bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])
                        # for j, c in enumerate(bitstring):
                        #     if c == '0':
                        #         bsign[j] = -1
                        #     else:
                        #         bsign[j] = 1
                        b = bsign

                        if self.weight_type == 'gamma': 
                            M = torch.randn(resnet_channels[layer_key][0], output_channels)
                        else:
                            M = torch.randn(resnet_channels[layer_key][1], output_channels)

                        passport_kwargs[layer_key][i][module_key]['b'] = b
                        passport_kwargs[layer_key][i][module_key]['M'] = M

        else:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            passport_kwargs[layer_key] = {
                'flag': flag
            }
            
            if b is not None:
                if model == 'alexnet':
                    if layer_key == "4":
                        output_channels = int (bit_length * 384 / 896)
                    if layer_key == "5":
                        output_channels = int (bit_length * 256/ 896)
                    if layer_key == "6":
                        output_channels = int (bit_length * 256/ 896)
                if model == 'cnn' and dataset == 'cifar100':
                    if layer_key == "0":
                        output_channels = int (bit_length / 3)
                    if layer_key == "2":
                        output_channels = int (bit_length * 2 / 3)                        
                if model == 'cnn' and dataset == 'fmnist' or dataset == 'mnist':
                    if layer_key == "0":
                        output_channels = int (bit_length / 2)
                    if layer_key == "2":
                        output_channels = int (bit_length / 2)                        
                if model == 'cnn' and dataset == 'cifar10':
                    if layer_key == "0":
                        output_channels = int (bit_length / 2)
                    if layer_key == "2":
                        output_channels = int (bit_length / 2)  

                #output_channels = len(b) * 8
                bsign = torch.sign(torch.rand(output_channels) - 0.5)
                # bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])
                              
                # for j, c in enumerate(bitstring):
                #     if c == '0':
                #         bsign[j] = -1
                #     else:
                #         bsign[j] = 1
                b = bsign
                if model == 'alexnet':
                    if self.weight_type == 'gamma': 
                        M = torch.randn(alexnet_channels[layer_key][0], output_channels)
                    else:
                        M = torch.randn(alexnet_channels[layer_key][1], output_channels)
                if model == 'cnn' and dataset == 'cifar100':
                    if self.weight_type == 'gamma': 
                        M = torch.randn(cnn_channels[layer_key][0], output_channels)
                    else:
                        M = torch.randn(cnn_channels[layer_key][1], output_channels) 
                if model == 'cnn' and dataset == 'fmnist':
                    if self.weight_type == 'gamma': 
                        M = torch.randn(mnist_channels[layer_key][0], output_channels)
                    else:
                        M = torch.randn(mnist_channels[layer_key][1], output_channels) 
                if model == 'cnn' and dataset == 'cifar10':
                    if self.weight_type == 'gamma': 
                        M = torch.randn(mnist_channels[layer_key][0], output_channels)
                    else:
                        M = torch.randn(mnist_channels[layer_key][1], output_channels) 


                passport_kwargs[layer_key]['b'] = b
                passport_kwargs[layer_key]['M'] = M

    return passport_kwargs
# 文件保存    
def to_csv(file, data):
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
