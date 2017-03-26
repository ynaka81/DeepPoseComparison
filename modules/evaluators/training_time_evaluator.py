# -*- coding: utf-8 -*-
""" Evaluate training time. """

import os
import matplotlib.pyplot as plt

from modules.evaluators import chainer, pytorch


class TrainingTimeEvaluator(object):
    """ Evaluate training time of pose net by chainer and pytorch.

    Args:
        chainer_log (str): Log file of chainer.
        pytorch_log (str): Log file of pytorch.
        output (str): Output directory.
    """

    def __init__(self,
                 chainer_log='result/chainer/log',
                 pytorch_log='result/pytorch/log',
                 output='result'):
        try:
            os.makedirs(output)
        except OSError:
            pass
        self.chainer_log = chainer.TrainingLog(chainer_log)
        self.pytorch_log = pytorch.TrainingLog(pytorch_log)
        self.output = output

    def plot(self, title, debug):
        """ Plot training time of chainer and pytorch. """
        # plot training time.
        plt.plot(self.chainer_log.t, self.chainer_log.v, label='chainer')
        plt.plot(self.pytorch_log.t, self.pytorch_log.v, label='pytorch')
        # plot settings.
        plt.title(title)
        plt.legend()
        plt.xlabel('(log scale) training time [sec]')
        plt.ylabel('loss function value')
        plt.xscale('log')
        # save plot.
        if debug:
            plt.show()
        else:
            filename = '_'.join(title.split(' ')) + '.png'
            plt.savefig(os.path.join(self.output, filename))
