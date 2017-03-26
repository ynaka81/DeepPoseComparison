# -*- coding: utf-8 -*-
""" Estimate pose by pytorch. """

import torch
from torch.autograd import Variable
from torchvision import transforms

from modules.errors import GPUNotFoundError
from modules.models.pytorch import AlexNet
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale


class PoseEstimator(object):
    """ Estimate pose using pose net trained by pytorch.

    Args:
        Nj (int): Number of joints.
        gpu (int): GPU ID (negative value indicates CPU).
        model_file (str): Model parameter file.
        filename (str): Image-pose list file.
    """

    def __init__(self, Nj, gpu, model_file, filename):
        # validate arguments.
        self.gpu = (gpu >= 0)
        if self.gpu and not torch.cuda.is_available():
            raise GPUNotFoundError('GPU is not found.')
        # initialize model to estimate.
        self.model = AlexNet(Nj)
        self.model.load_state_dict(torch.load(model_file))
        # prepare gpu.
        if self.gpu:
            self.model.cuda()
        # load dataset to estimate.
        self.dataset = PoseDataset(
            filename,
            input_transform=transforms.Compose([
                transforms.ToTensor(),
                RandomNoise()]),
            output_transform=Scale(),
            transform=Crop(data_augmentation=True))

    def get_dataset_size(self):
        """ Get size of dataset. """
        return len(self.dataset)

    def estimate(self, index):
        """ Estimate pose of i-th image. """
        image, _, _ = self.dataset[index]
        v_image = Variable(image.unsqueeze(0))
        if self.gpu:
            v_image = v_image.cuda()
        self.model.forward(v_image)
