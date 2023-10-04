import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *
import math




def SelectModel(m):
    
    if m == 'MobilenetV2':
        cfg=[[1, 16, 1, 1],[6, 24, 2, 2], [6, 32, 3, 2],
              [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
              [6, 320, 1, 1]]
        return MobileNetV2(MobileBottleneck, cfg) 


class MobileNetV2(nn.Module):
    def __init__(self, block, cfg, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.in_planes = 32
        self.conv1 = Conv(3, 32, kernel_size=3, stride=2, padding=1)
        self.layer = self._make_layer_(block, cfg)
        self.conv2 = Conv(320, 1280, kernel_size=1, stride=1)

        self.linear = nn.Linear(1280, num_classes)
        self._initialize_weights()

    def _make_layer_(self, block, cfg):
        layers = []
        for i in range(len(cfg)):
            param = cfg[i]
            planes = param[1]
            t = param[0]
            stride = param[3]
            num_blocks = param[2]
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride, t))
                self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        o = self.conv1(x)
        o = self.layer(o)
        o = self.conv2(o)
        o = F.avg_pool2d(o, o.size()[3])
        o = o.view(o.size(0), -1)
        o = self.linear(o)
        return o

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()