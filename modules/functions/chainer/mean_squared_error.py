# -*- coding: utf-8 -*-
""" Mean squared error function. """

import numpy as np
from chainer import function
from chainer.utils import type_check


class MeanSquaredError(function.Function):
    """ Mean squared error (a.k.a. Euclidean loss) function. """
    def __init__(self, use_visibility=False):
        self.use_visibility = use_visibility
        self.diff = None
        self.N = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.int32,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape[:-1] == in_types[2].shape[:-1]
        )

    def forward_cpu(self, inputs):
        x, t, v = inputs
        self.diff = x - t
        if self.use_visibility:
            self.N = v.sum()/2
            self.diff *= v
        else:
            self.N = self.diff.size/2
        diff = self.diff.ravel()
        return np.array(diff.dot(diff)/self.N, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x, t, v = inputs
        self.diff = x - t
        if self.use_visibility:
            self.N = int(v.sum())/2
            self.diff *= v
        else:
            self.N = self.diff.size/2
        diff = self.diff.ravel()
        return diff.dot(diff)/diff.dtype.type(self.N),

    def backward(self, inputs, gy):
        coeff = gy[0]*gy[0].dtype.type(2./self.N)
        gx0 = coeff*self.diff
        return gx0, -gx0, None


def mean_squared_error(x, t, v, use_visibility=False):
    """ Computes mean squared error over the minibatch.

    Args:
        x (Variable): Variable holding an float32 vector of estimated pose.
        t (Variable): Variable holding an float32 vector of ground truth pose.
        v (Variable): Variable holding an int32 vector of ground truth pose's visibility.
            (0: invisible, 1: visible)
        use_visibility (bool): When it is ``True``,
            the function uses visibility to compute mean squared error.
    Returns:
        Variable: A variable holding a scalar of the mean squared error loss.
    """
    return MeanSquaredError(use_visibility)(x, t, v)
