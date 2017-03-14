# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_
import torch
from torch.autograd import Variable

from modules.functions.pytorch import mean_squared_error


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        self.x = torch.rand(10, 14, 2)
        self.t = torch.rand(10, 14, 2)
        self.v = torch.bernoulli(torch.rand(10, 14, 1)).expand(10, 14, 2).byte()

    def check_forward(self, x_data, t_data, v_data, use_visibility):
        x = Variable(x_data)
        t = Variable(t_data)
        v = Variable(v_data)
        loss = mean_squared_error(x, t, v, use_visibility)
        loss_data = loss.data
        eq_(type(loss_data), torch.FloatTensor)
        eq_(loss_data.size(), (1,))
        # compute expected value.
        loss_expect = 0.
        N = 0
        for x_i, t_i, v_i in zip(x.view(-1), t.view(-1), v.view(-1)):
            if use_visibility and (v_i.data == 0).all():
                continue
            diff = t_i - x_i
            loss_expect += diff**2
            N += 1
        loss_expect /= N
        self.assertAlmostEqual(loss_expect, loss, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.v, False)
        self.check_forward(self.x, self.t, self.v, True)
