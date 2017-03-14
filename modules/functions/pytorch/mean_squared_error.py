# -*- coding: utf-8 -*-
""" Mean squared error function. """

import torch
import torch.nn as nn


class MeanSquaredError(nn.Module):
    """ Mean squared error (a.k.a. Euclidean loss) function. """

    def __init__(self, use_visibility=False):
        super(MeanSquaredError, self).__init__()
        self.use_visibility = use_visibility

    # pylint: disable=arguments-differ
    def forward(self, *inputs):
        x, t, v = inputs
        if self.use_visibility:
            X = torch.masked_select(x, v)
            T = torch.masked_select(t, v)
        else:
            X = x.view(-1)
            T = t.view(-1)
        diff = T - X
        return diff.dot(diff)/diff.numel()

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
