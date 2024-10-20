import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # Убедитесь, что указаны размеры ядра
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # Убедитесь, что указаны размеры ядра

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # Размер после двух сверток и двух пулингов
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Реализация forward pass сети
        x = self.pool1(F.relu(self.conv1(x)))  # conv1 -> maxpool1
        x = self.pool2(F.relu(self.conv2(x)))  # conv2 -> maxpool2
        x = self.flatten(x)                     # flatten
        x = F.relu(self.fc1(x))                # fc1
        x = self.fc2(x)                        # fc2
        return x

def create_model():
    return ConvNet()
