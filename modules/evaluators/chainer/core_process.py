# -*- coding: utf-8 -*-
""" Core process (forward + backward) of chainer. """

import chainer
from chainer import serializers

from modules.dataset_indexing.chainer import PoseDataset
from modules.models.chainer import AlexNet


class CoreProcess(object):
    """ Core process (forward + backward) of training by chainer.

    Args:
        batch_size (int): Batch size of each iteration.
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        model_file (str): Model parameter file.
        filename (str): Image-pose list file.
    """

    def __init__(self, Nj, gpu, model_file, filename):
        # initialize model to estimate.
        self.model = AlexNet(Nj, use_visibility=True)
        self.gpu = gpu
        serializers.load_npz(model_file, self.model)
        # prepare gpu.
        if self.gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu()
        # load dataset to estimate.
        self.dataset = PoseDataset(filename)

    def set_batch_size(self, batch_size):
        """ Set batch size of core process. """
        self.iter = chainer.iterators.MultiprocessIterator(self.dataset, batch_size)

    def run(self, only_inference):
        """ Run core process. """
        batch = self.iter.next()
        in_arrays = chainer.dataset.convert.concat_examples(batch, self.gpu)
        in_vars = tuple(chainer.Variable(x) for x in in_arrays)
        if only_inference:
            self.model.predict(in_vars[0])
        else:
            y = self.model(*in_vars)
            y.backward()
