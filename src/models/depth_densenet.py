"""DepthDensenet in PyTorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthBottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, use_bn=True):
        super(DepthBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=growth_rate,
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if hasattr(self, "bn2"):
            out = self.conv2(F.relu(self.bn2(out)))
        else:
            out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


class DepthTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DepthTransition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DepthDensenet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DepthDensenet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(
            block, num_planes, nblocks[0], use_bn=False
        )
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = DepthTransition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = DepthTransition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(
            block, num_planes, nblocks[2], use_bn=False
        )
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = DepthTransition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, use_bn=True):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, use_bn))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def depth_densenet_cifar():
    return DepthDensenet(
        DepthBottleneck, [6, 12, 24, 16], growth_rate=12, reduction=0.5
    )
