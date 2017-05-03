# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch
from torch.autograd import Function


class MeanSquaredError(Function):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False):
        super(MeanSquaredError, self).__init__()
        self.use_visibility = use_visibility

    def forward(self, *inputs):
        x, t, v = inputs
        self.diff = x - t
        if self.use_visibility:
            self.N = v.sum()/2
            self.diff = self.diff*v
        else:
            self.N = self.diff.numel()/2
        diff = self.diff.view(-1)
        return torch.Tensor([diff.dot(diff)/self.N])

    def backward(self, *grad_outputs):
        coeff = grad_outputs[0][0]*2/self.N
        gx0 = coeff*self.diff
        return gx0, None, None


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
