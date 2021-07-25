import os
import torch
import pandas as pd
import numpy as np
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, n):
        super(DenseBlock, self).__init__()
        inplanes = ((n)*3 + 2) * 13
        self.add_module('bn1', nn.BatchNorm2d(inplanes))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(inplanes, (n+1)*13, kernel_size=1, stride=1))
        self.add_module('bn2', nn.BatchNorm2d((n+1)*13))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d((n+1)*13, 39, kernel_size=3, stride=1,padding=1)) # Must add padding
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
    
    def forward(self,x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # out += identity
        out = torch.cat((out, identity), dim=1)

        out = self.pool(out)

        return out

class NeuralNetwork(nn.Module):
    def __init__(self, num_dense):
        super(NeuralNetwork, self).__init__()
        self.num_dense = num_dense
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=26, kernel_size=3),
            nn.MaxPool2d(2, stride=2))
        self.end = nn.Sequential(
            nn.BatchNorm2d((2+3*5)*13),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        for i in range(self.num_dense):
            block = DenseBlock(i)
            self.initial.add_module('denseblock%d' % (i + 1), block)

        # not sure what the input dimension is
        self.fc = nn.Linear(221, 1)

    def forward(self, x):
        out = self.initial(x)

        out = self.end(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out