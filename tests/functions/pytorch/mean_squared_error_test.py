# -*- coding: utf-8 -*-

import unittest
from nose.tools import eq_, ok_
import torch
from torch.autograd import Variable, gradcheck

from modules.functions.pytorch import mean_squared_error
from modules.functions.pytorch.mean_squared_error import MeanSquaredError


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        self.x = torch.rand(10, 14, 2)
        self.t = torch.rand(10, 14, 2)
        self.v = torch.bernoulli(torch.rand(10, 14, 1)).expand(10, 14, 2).clone()

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
        N /= 2
        loss_expect /= N
        self.assertAlmostEqual(loss_expect, loss, places=5)

    def test_forward(self):
        self.check_forward(self.x, self.t, self.v, False)
        self.check_forward(self.x, self.t, self.v, True)

    def check_backward(self, x_data, t_data, v_data, use_visibility):
        x = Variable(x_data, requires_grad=True)
        t = Variable(t_data)
        v = Variable(v_data)
        test = gradcheck(
            MeanSquaredError(use_visibility),
            (x, t, v), eps=1e-2, atol=1e-3)
        ok_(test)

    def test_bakward(self):
        self.check_backward(self.x, self.t, self.v, False)
        self.check_backward(self.x, self.t, self.v, True)
