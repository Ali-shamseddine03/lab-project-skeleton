
import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 200)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = self.pool(self.conv3(x).relu())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        return x
