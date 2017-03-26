# -*- coding: utf-8 -*-
""" Evaluate estimating time. """

import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from modules.evaluators import chainer, pytorch


class EstimatingTimeEvaluator(object):
    """ Evaluate estimating time of pose net by chainer and pytorch.

    Args:
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        chainer_model_file (str): Chainer model parameter file.
        pytorch_model_file (str): Pytorch model parameter file.
        filename (str): Image-pose list file.
        output (str): Output directory.
        debug (bool): Debug mode.
    """

    def __init__(self, **kwargs):
        self.output = kwargs['output']
        try:
            os.makedirs(self.output)
        except OSError:
            pass
        self.estimator = {
            'chainer': chainer.PoseEstimator(
                kwargs['Nj'], kwargs['gpu'], kwargs['chainer_model_file'], kwargs['filename']),
            'pytorch': pytorch.PoseEstimator(
                kwargs['Nj'], kwargs['gpu'], kwargs['pytorch_model_file'], kwargs['filename'])}
        self.debug = kwargs['debug']

    def plot(self, samples, title):
        """ Plot estimating time of chainer and pytorch. """
        time_mean = []
        time_std = []
        for estimator in tqdm(self.estimator.values(), desc='testers'):
            random_index = np.random.randint(0, estimator.get_dataset_size(), samples)
            compute_time = []
            # get compute time.
            for index in tqdm(random_index, desc='samples'):
                start = time.time()
                estimator.estimate(index)
                compute_time.append(time.time() - start)
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
        if self.debug:
            plt.show()
        else:
            filename = '_'.join(title.split(' ')) + '.png'
            plt.savefig(os.path.join(self.output, filename))
