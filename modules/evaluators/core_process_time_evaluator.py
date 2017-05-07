# -*- coding: utf-8 -*-
""" Evaluate core process (forward + backward) time. """

import os
import time
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

from modules.evaluators import chainer, pytorch


class CoreProcessTimeEvaluator(object):
    """ Evaluate core process (forward + backward) time of pose net by chainer and pytorch.

    Args:
        max_batch_index (int): Index of max batch size: 2^m.
        only-inference (bool): If true, core process includes only inference process.
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
        self.process = {
            'chainer': chainer.CoreProcess(kwargs['Nj'], kwargs['gpu'], kwargs['chainer_model_file'], kwargs['filename']),
            'pytorch': pytorch.CoreProcess(kwargs['Nj'], kwargs['gpu'], kwargs['pytorch_model_file'], kwargs['filename'])}
        self.max_batch_index = kwargs['max_batch_index']
        self.only_inference = kwargs['only_inference']
        self.debug = kwargs['debug']

    def plot(self, samples, title):
        """ Plot core process time of chainer and pytorch. """
        batch_sizes = [2**m for m in range(self.max_batch_index + 1)]
        process_time = {'chainer': {'mean': [], 'std': []},
                        'pytorch': {'mean': [], 'std': []}}
        for batch_size in tqdm(batch_sizes, desc='batch size'):
            for name, process in tqdm(self.process.items(), desc='testers'):
                # set batch size.
                process.set_batch_size(batch_size)
                compute_time = []
                # get compute time.
                for index in trange(samples, desc='samples'):
                    start = time.time()
                    process.run(self.only_inference)
                    compute_time.append(time.time() - start)
                # calculate mean and std.
                process_time[name]['mean'].append(np.mean(compute_time))
                process_time[name]['std'].append(np.std(compute_time))
        # plot core process time of each batch size.
        for name, p_t in process_time.items():
            plt.errorbar(batch_sizes, p_t['mean'], yerr=p_t['std'], label=name)
        # plot settings.
        plt.title(title)
        plt.legend(loc='lower right')
        plt.xlabel('batch size')
        plt.ylabel('core process time [sec]')
        # save plot.
        if self.debug:
            plt.show()
        else:
            filename = '_'.join(title.split(' ')) + '.png'
            plt.savefig(os.path.join(self.output, filename))
