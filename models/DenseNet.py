import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_channels = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for i in range(len(nblocks)-1):
            self.features.add_module('dense_block_{}'.format(i+1), self._make_dense_layers(block, num_channels, nblocks[i]))
            num_channels += nblocks[i]*growth_rate
            out_channels = int(num_channels*reduction)
            self.features.add_module('transition_{}'.format(i+1), Transition(num_channels, out_channels))
            num_channels = out_channels

        self.features.add_module('dense_block_{}'.format(len(nblocks)), self._make_dense_layers(block, num_channels, nblocks[-1]))
        num_channels += nblocks[-1]*growth_rate

        self.features.add_module('bn', nn.BatchNorm2d(num_channels))
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_dense_layers(self, block, in_channels, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet_cifar(num_classes):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,num_classes=num_classes)