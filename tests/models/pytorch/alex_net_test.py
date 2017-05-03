# -*- coding: utf-8 -*-

import unittest
from nose.tools import eq_
import torch
from torch.autograd import Variable

from modules.models.pytorch import AlexNet


class TestAlexNet(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.Nj = 14
        self.images = Variable(torch.rand(self.N, 3, 227, 227))
        self.model = AlexNet(self.Nj)

    def test_forward(self):
        pose = self.model.forward(self.images)
        eq_(pose.size(), (self.N, self.Nj, 2))
