# -*- coding: utf-8 -*-
""" AlexNet implementation. """

import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """ The AlexNet :
    'A. Krizhevsky, I. Sutskever, and G. Hinton.
    Imagenet clas-sification with deep convolutional neural networks. InNIPS , 2012'

    Args:
        Nj (int): Size of joints.
    """

    def __init__(self, Nj):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, Nj*2)
        self.Nj = Nj

    def forward(self, x):
        # layer1
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, 3, stride=2)
        # layer2
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, 3, stride=2)
        # layer3-5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pool2d(h, 3, stride=2)
        h = h.view(-1, 256*6*6)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), training=self.training)
        h = F.dropout(F.relu(self.fc7(h)), training=self.training)
        h = self.fc8(h)
        return h.view(-1, self.Nj, 2)
