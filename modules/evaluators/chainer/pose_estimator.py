# -*- coding: utf-8 -*-
""" Estimate pose by chainer. """

import numpy as np
import chainer
from chainer import cuda, serializers

from modules.dataset_indexing.chainer import PoseDataset
from modules.models.chainer import AlexNet


class PoseEstimator(object):
    """ Estimate pose using pose net trained by chainer.

    Args:
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        model_file (str): Model parameter file.
        filename (str): Image-pose list file.
    """

    def __init__(self, Nj, gpu, model_file, filename):
        # initialize model to estimate.
        self.model = AlexNet(Nj)
        self.gpu = gpu
        serializers.load_npz(model_file, self.model)
        # prepare gpu.
        if self.gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu()
        # load dataset to estimate.
        self.dataset = PoseDataset(filename)

    def get_dataset_size(self):
        """ Get size of dataset. """
        return len(self.dataset)

    def estimate(self, index):
        """ Estimate pose of i-th image. """
        image, _, _ = self.dataset.get_example(index)
        v_image = np.array([image])
        if self.gpu >= 0:
            v_image = cuda.to_gpu(v_image)
        self.model.predict(v_image)
