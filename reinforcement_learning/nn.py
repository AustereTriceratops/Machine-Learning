import torch
import numpy as np

from constants import *
from utils import *



class Res2d(torch.nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        k = int((kernel_size-1)/2)
        k_ = k
        if kernel_size % 2 == 0:
            k_ += 1

        self.pad = torch.nn.ZeroPad2d((k, k_, k, k_))
        self.bn = torch.nn.BatchNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x_ = self.conv1(self.pad(self.activation(self.bn(x))))
        x_ = self.conv1(self.pad(self.activation(self.bn(x_))))

        return x_ + x 


class ResNet(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, depth=1):
        super().__init__()

        self.depth = depth
        self.layers = torch.nn.ModuleList([Res2d(channels, kernel_size) for _ in range(depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
        return x


class ActionHead(torch.nn.Module):
    def __init__(self, in_channels, phase):
        super().__init__()

        self.activation = torch.nn.ReLU()
        self.bn_1 = torch.nn.BatchNorm2d(in_channels)
        self.bn_2 = torch.nn.BatchNorm1d(150)

        self.downconv = torch.nn.Conv2d(in_channels, 6, kernel_size=3, padding=(1,1))
        self.flatten = torch.nn.Flatten(start_dim=1)

        if phase == "moving":
            self.fc = torch.nn.Linear(150, 200)
        elif phase == "placing":
            self.fc = torch.nn.Linear(150, 25)


    def forward(self, x):
        x_ = self.downconv(self.activation(self.bn_1(x)))
        x_ = self.flatten(x_)
        x_ = self.fc(self.activation(self.bn_2(x_)))

        return x_


class Network(torch.nn.Module):
    def __init__(self, temperature=1):
        super().__init__()

        self.temperature = temperature

        self.entry_net = torch.nn.Conv2d(3, 16, kernel_size=3, padding=(1,1))
        self.resnet = ResNet(channels=16, kernel_size=3, depth=4)
        self.value_head = torch.nn.Conv2d(16, 1, kernel_size=5)
        self.placing_head = ActionHead(16, phase="placing")
        self.moving_head = ActionHead(16, phase="moving")

        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, prompt):
        # x is (*, 3, 5, 5) tensor
        # think about how many channels are actually used

        x = self.activation(self.entry_net(x)) # x is now (*, 16, 5, 5)

        x = self.resnet(x)

        value_ = self.value_head(self.activation(self.bn(x)))  # value is (*, 1) tensor
        value_ = self.flatten(value_)
        value = self.sigmoid(value_)

        action = None

        if prompt == "placing":
            action = self.placing_head(x)  # action is (*, 25) 
        elif prompt == "moving":
            action = self.moving_head(x)   # or a (*, 200) tensor


        #action *= self.temperature
        #action = self.softmax(action)

        return value, action
