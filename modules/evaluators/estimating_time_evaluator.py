# -*- coding: utf-8 -*-
""" Evaluate estimating time. """

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from modules.evaluators import chainer, pytorch


class EstimatingTimeEvaluator(object):
    """ Evaluate estimating time of pose net by chainer and pytorch.

    Args:
        batchsize (int): Estimating batch size.
        output (str): Output directory.
    """

    def __init__(self, batchsize=1, output='result'):
        try:
            os.makedirs(output)
        except OSError:
            pass
        self.estimator = {'chainer': chainer.PoseEstimator(batchsize),
                          'pytorch': pytorch.PoseEstimator(batchsize)}
        self.output = output

    def plot(self, title, debug):
        """ Plot estimating time of chainer and pytorch. """
        time_mean = []
        time_std = []
        for estimator in self.estimator.values():
            compute_time = []
            # get compute time.
            end = time.time()
            for _ in estimator.get_pose_list():
                compute_time.append(time.time() - end)
                end = time.time()
            # calculate mean and std.
            time_mean.append(np.mean(compute_time))
            time_std.append(np.std(compute_time))
        # plot estimating time.
        plt.bar(range(len(time_mean)), time_mean, yerr=time_std,
                width=0.3, align='center', ecolor='black', tick_label=self.estimator.keys())
        # plot settings.
        plt.title(title)
        plt.ylabel('estimating time [sec]')
        # save plot.
        if debug:
            plt.show()
        else:
            filename = '_'.join(title.split(' ')) + '.png'
            plt.savefig(os.path.join(self.output, filename))
