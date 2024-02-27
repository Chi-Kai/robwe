import torch
import torch.nn as nn
from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d_private import PassportPrivateBlock

class AlexNet(nn.Module):

    def __init__(self, in_channels, num_classes, passport_settings):
        super().__init__()

        passport_kwargs = {}
        for layer_key in passport_settings:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            passport_kwargs[layer_key] = {
                'flag': flag
            }
            maxpoolidx = [1, 3, 7]
        layers = []
        head_layers = []
        inp = in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                if passport_kwargs[str(layeridx)]['flag']:
                    layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p))
                    head_layers.append(PassportPrivateBlock(inp, oups[layeridx], k, 1, p).weight)
                else:
                    layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4 * 4 * 256, num_classes)
        self.weight_keys = [
            ['features[0].weight', 'features[0].bias'],
            ['features[2].weight', 'features[2].bias'],
            ['features[4].weight', 'features[4].bias'],
            ['features[5].weight', 'features[5].bias'],
            ['features[6].weight', 'features[6].bias'],
            ['classifier.weight', 'classifier.bias']
        ]
    def forward(self, x):
        for m in self.features:
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    #返回所有fc层权重,放到一个list里
    def head_params(self):
        return self.head_layers
    #返回head层的参数
    def get_params(self):
        pass
    # 返回表示层的参数
    def get_rep_params(self):
        pass
    # 所有表示层参数
    def rep_params(self):
        return [self.classifier.weight]