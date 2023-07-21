import torch
import torch.nn as nn

class CustomResNet01(nn.Module):
    def __init__(self):
        super(CustomResNet01, self).__init__()
        
        '''PrepLayer'''
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        '''Layer-1'''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        '''Max Pool'''
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        '''ResBlock-1'''
        self.resBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        '''Layer-2'''
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        '''Max Pool'''
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        '''Layer-3'''
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        '''Max Pool'''
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=2)
        '''ResBlock-2'''
        self.resBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        '''Average Pool'''
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        '''FC Layer'''
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.prepLayer(x)
        x = self.layer1(x)
        x = self.maxpool1(x)

        residualBlock1 = self.resBlock1(x)
        x = x + residualBlock1

        x = self.layer2(x)
        x = self.maxpool2(x)

        x = self.layer3(x)
        x = self.maxpool3(x)
        residualBlock2 = self.resBlock2(x)
        x = x + residualBlock2

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x