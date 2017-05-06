# -*- coding: utf-8 -*-
""" Evaluate training time. """

import os
import matplotlib.pyplot as plt

from modules.evaluators import chainer, pytorch


class TrainingTimeEvaluator(object):
    """ Evaluate training time of pose net by chainer and pytorch.

    Args:
        chainer_logs (str): Log files of chainer.
        pytorch_logs (str): Log files of pytorch.
        output (str): Output directory.
    """

    def __init__(self,
                 chainer_logs=[],
                 pytorch_logs=[],
                 output='result'):
        try:
            os.makedirs(output)
        except OSError:
            pass
        self.logs = {'chainer': map(lambda log: chainer.TrainingLog(log), chainer_logs),
                     'pytorch': map(lambda log: pytorch.TrainingLog(log), pytorch_logs)}
        self.legends = dict(map(lambda key: (key, [key]), self.logs.keys()))
        for name, logs in zip(('chainer', 'pytorch'), (chainer_logs, pytorch_logs)):
            if len(logs) > 1:
                self.legends[name] = map(lambda log: '{}/{}'.format(name, os.path.basename(log)), logs)
        self.output = output

    def plot(self, title, debug):
        """ Plot training time of chainer and pytorch. """
        # plot training time.
        for name in self.logs.keys():
            for log, legend in zip(self.logs[name], self.legends[name]):
                plt.plot(log.t, log.v, label=legend)
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
