#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:02:53 2022

@author: qzhang
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_MD(nn.Module):
    def __init__(self,num_modes):
        super(VGG16_MD, self).__init__()
        # conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)

        self.conv1_1 = nn.Conv2d(1,32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.maxpool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 1024)
        self.fc8 = nn.Linear(1024, num_modes*2-1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
       # x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        # x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc6(x))
        # x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc7(x))
        # x = F.dropout(x, 0.5)
        x = self.fc8(x)
        # x = F.hardtanh(x,0,1)
        return x
        