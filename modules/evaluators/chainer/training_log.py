# -*- coding: utf-8 -*-
""" Training log of chainer. """

import json


class TrainingLog(object):
    """ Training log of pose net by chainer.

    Args:
        log (str): Log file.
    """

    def __init__(self, log):
        self.t = []
        self.v = []
        self._load(log)

    def _load(self, log):
        for data in json.load(open(log)):
            self.t.append(data['elapsed_time'])
            self.v.append(data['main/loss'])
