# -*- coding: utf-8 -*-
# pylint: skip-file
""" Estimate pose by pytorch. """

import time


class PoseEstimator(object):
    """ Estimate pose using pose net trained by pytorch.

    Args:
        batchsize (int): Estimating batch size.
    """

    def __init__(self, batchsize):
        # TODO: implement
        pass

    def get_pose_list(self):
        """ Get estimated pose list. """
        # TODO: implement
        time.sleep(0.07)
        yield 1
        time.sleep(0.1)
        yield 2
        time.sleep(0.12)
        yield 3
