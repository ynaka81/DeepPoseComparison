# -*- coding: utf-8 -*-
# pylint: skip-file
""" Estimate pose by chainer. """

import time


class PoseEstimator(object):
    """ Estimate pose using pose net trained by chainer.

    Args:
        batchsize (int): Estimating batch size.
    """

    def __init__(self, batchsize):
        # TODO: implement
        pass

    def get_pose_list(self):
        """ Get estimated pose list. """
        # TODO: implement
        time.sleep(0.1)
        yield 1
        time.sleep(0.2)
        yield 2
        time.sleep(0.15)
        yield 3
