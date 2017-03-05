# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_
import numpy as np
import chainer
from chainer.testing import condition

from modules.models.chainer import AlexNet


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        self.N = 3
        self.Nj = 14
        self.images = np.random.uniform(0, 1, (self.N, 3, 55, 51)).astype(np.float32)
        self.x = np.random.uniform(0, 1, (self.N, self.Nj, 2)).astype(np.float32)
        self.v = np.random.randint(0, 2, (self.N, self.Nj, 1)).astype(np.int32)

    @condition.repeat(5)
    def test_predict(self):
        for use_visibility in (False, True):
            model = AlexNet(self.Nj, use_visibility=use_visibility)
            pose = model.predict(self.images)
            eq_(pose.shape, (self.N, self.Nj, 2))

    @condition.repeat(5)
    def test_call(self):
        for use_visibility in (False, True):
            model = AlexNet(self.Nj, use_visibility=use_visibility)
            x = chainer.Variable(self.x)
            v = chainer.Variable(self.v)
            model(self.images, x, v)
