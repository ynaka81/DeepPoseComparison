# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch.nn as nn


class MeanSquaredError(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False):
        super(MeanSquaredError, self).__init__()
        self.use_visibility = use_visibility

    def forward(self, *inputs):
        x, t, v = inputs
        diff = x - t
        if self.use_visibility:
            N = (v.sum()/2).data[0]
            diff = diff*v
        else:
            N = diff.numel()/2
        diff = diff.view(-1)
        return diff.dot(diff)/N


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
