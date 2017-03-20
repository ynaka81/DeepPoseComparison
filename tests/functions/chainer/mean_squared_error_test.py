# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_, nottest
import numpy as np
import chainer
from chainer import cuda
from chainer import gradient_check
from chainer.testing import attr
from chainer.testing import condition

from modules.functions.chainer import mean_squared_error
from modules.functions.chainer.mean_squared_error import MeanSquaredError


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        self.x = np.random.uniform(0, 1, (10, 14, 2)).astype(np.float32)
        self.t = np.random.uniform(0, 1, (10, 14, 2)).astype(np.float32)
        self.v = np.random.randint(0, 2, (10, 14, 1)).astype(np.int32)

    def check_forward(self, x_data, t_data, v_data, use_visibility):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        v = chainer.Variable(v_data)
        loss = mean_squared_error(x, t, v, use_visibility)
        loss_value = cuda.to_cpu(loss.data)
        eq_(loss_value.dtype, np.float32)
        eq_(loss_value.shape, ())
        # compute expected value.
        loss_expect = 0.
        for i in np.ndindex(self.x.shape):
            diff = self.x[i] - self.t[i]
            if use_visibility:
                diff *= self.v[i[:-1]]
            loss_expect += diff**2
        if use_visibility:
            N = self.v.sum()/2
        else:
            N = self.x.size/2
        loss_expect /= N
        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.repeat(10)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t, self.v, False)
        self.check_forward(self.x, self.t, self.v, True)

    @nottest
    @attr.gpu
    @condition.repeat(10)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.v), False)
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.v), True)

    def check_backward(self, x, t, v, use_visibility):
        gradient_check.check_backward(
            MeanSquaredError(use_visibility),
            (x, t, v), None, eps=1e-2,
            no_grads=(False, False, True))

    @condition.repeat(10)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t, self.v, False)
        self.check_backward(self.x, self.t, self.v, True)

    @nottest
    @attr.gpu
    @condition.repeat(10)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.v), False)
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), cuda.to_gpu(self.v), True)
