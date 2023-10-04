import torch
import torch.nn as nn
import torch.nn.functional as F




def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



class Conv(nn.Module):
    def __init__(self, input, output, kernel_size, stride, padding=0, groups=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input, output, kernel_size, stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MobileBottleneck(nn.Module):
    def __init__(self, input, output, stride, t):
        super(MobileBottleneck, self).__init__()
        self.stride = stride
        self.resent = self.stride == 1 and input == output
        h = int(t*input)
        if t == 1:
            self.conv = nn.Sequential(
                Conv(input, h, kernel_size=3, stride=stride, padding=1, groups=h),
                nn.Conv2d(h, output, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(output)
            )
        else:
            self.conv = nn.Sequential(
                Conv(input, h, kernel_size=1, stride=1),
                Conv(h, h, kernel_size=3, stride=stride, padding=1, groups=h),
                nn.Conv2d(h, output, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(output)
            )


    def forward(self, x):
        if self.resent:
            o = x + self.conv(x)
        else:
            o = self.conv(x)
        return o