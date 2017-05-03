# -*- coding: utf-8 -*-
""" Training log of pytorch. """


class TrainingLog(object):
    """ Training log of pose net by pytorch.

    Args:
        log (str): Log file.
    """

    def __init__(self, log):
        self.t = []
        self.v = []
        self._load(log)

    def _load(self, log):
        for line in open(log):
            split_line = line.split(', ')
            if split_line[1].startswith('loss'):
                elapsed_time = float(split_line[0].split(': ')[1])
                loss = float(split_line[1].split(': ')[1])
                self.t.append(elapsed_time)
                self.v.append(loss)
