# -*- coding: utf-8 -*-
""" Core process (forward + backward) of pytorch. """

import torch
from torch.autograd import Variable
from torchvision import transforms

from modules.errors import GPUNotFoundError
from modules.models.pytorch import AlexNet
from modules.dataset_indexing.pytorch import PoseDataset, Crop, RandomNoise, Scale
from modules.functions.pytorch import mean_squared_error


class CoreProcess(object):
    """ Core process (forward + backward) of training by pytorch.

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
            transform=Crop())

    def set_batch_size(self, batch_size):
        """ Set batch size of core process. """
        self.iter = iter(torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=True))

    def run(self, only_inference=False):
        """ Run core process. """
        batch_size = self.iter.batch_size
        try:
            batch = self.iter.next()
            if len(batch[0]) < batch_size:
                remain_batch = iter(torch.utils.data.DataLoader(self.dataset, batch_size - len(batch[0]), shuffle=True)).next()
                batch = map(lambda x: torch.cat(x), zip(batch, remain_batch))
        except StopIteration:
            # If self.iter arrive at the end of dataset, create new iter.
            self.iter = iter(torch.utils.data.DataLoader(self.dataset, batch_size, shuffle=True))
            batch = self.iter.next()
        in_vars = tuple(Variable(x) for x in batch)
        if self.gpu:
            in_vars = map(lambda x: x.cuda(), in_vars)
        y = self.model.forward(in_vars[0])
        if only_inference:
            return
        loss = mean_squared_error(y, in_vars[1], in_vars[2], use_visibility=True)
        loss.backward()
