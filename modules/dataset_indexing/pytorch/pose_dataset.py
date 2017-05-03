# -*- coding: utf-8 -*-
""" Pose dataset indexing. """

from PIL import Image
import torch
from torch.utils import data


class PoseDataset(data.Dataset):
    """ Pose dataset indexing.

    Args:
        path (str): A path to dataset.
        input_transform (Transform): Transform to input.
        output_transform (Transform): Transform to output.
        transform (Transform): Transform to both input and target.
    """

    def __init__(self, path, input_transform=None, output_transform=None, transform=None):
        self.path = path
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.transform = transform
        # load dataset.
        self.images, self.poses, self.visibilities = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """ Returns the i-th example. """
        image = self._read_image(self.images[index])
        pose = self.poses[index]
        visibility = self.visibilities[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.transform is not None:
            image, pose, visibility = self.transform(image, pose, visibility)
        if self.output_transform is not None:
            pose = self.output_transform(pose)
        return image, pose, visibility

    def _load_dataset(self):
        images = []
        poses = []
        visibilities = []
        for line in open(self.path):
            line_split = line[:-1].split(',')
            images.append(line_split[0])
            x = torch.Tensor(map(float, line_split[1:]))
            x = x.view(-1, 3)
            pose = x[:, :2]
            visibility = x[:, 2].clone().view(-1, 1).expand_as(pose)
            poses.append(pose)
            visibilities.append(visibility)
        return images, poses, visibilities

    @staticmethod
    def _read_image(path):
        return Image.open(path).convert('RGB')
