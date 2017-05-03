# -*- coding: utf-8 -*-
""" Transform input and outpupt data. """

import random
import numpy as np
import torch


class Crop(object):
    """ Crops the given PIL.Image to have a region of the given size.

    Args:
        data_augmentation (bool): True for data augmentation.
        crop_size (int): Size of cropping.
    """

    def __init__(self, data_augmentation=True, crop_size=227):
        self.data_augmentation = data_augmentation
        self.crop_size = crop_size

    def __call__(self, image, pose, visibility):
        _, height, width = image.size()
        shape = (width, height)
        visible_pose = torch.masked_select(pose, visibility.byte()).view(-1, 2)
        p_min = visible_pose.min(0)[0].squeeze()
        p_max = visible_pose.max(0)[0].squeeze()
        p_c = (p_min + p_max)/2
        crop_shape = [0, 0, 0, 0]
        # crop on a joint center
        for i in range(2):
            if self.data_augmentation:
                crop_shape[2*i] = random.randint(0, int(min(p_min[i], shape[i] - self.crop_size)))
            else:
                crop_shape[2*i] = max(0, int(p_c[i] - float(self.crop_size)/2))
            crop_shape[2*i + 1] = min(shape[i], crop_shape[2*i] + self.crop_size)
            crop_shape[2*i] -= self.crop_size - (crop_shape[2*i + 1] - crop_shape[2*i])
        transformed_image = image[:, crop_shape[2]:crop_shape[3], crop_shape[0]:crop_shape[1]]
        p_0 = torch.Tensor((crop_shape[0], crop_shape[2])).view(1, 2).expand_as(pose)
        transformed_pose = pose - p_0
        return transformed_image, transformed_pose, visibility


class RandomNoise(object):
    """ Give random noise to the given PIL.Image.
    """

    def __call__(self, image):
        numpy_image = image.numpy()
        # add random noise to keep eigen value.
        C = np.cov(np.reshape(numpy_image, (3, -1)))
        C = (C + C.T)/2
        l, e = np.linalg.eig(C)
        l = np.maximum(l, 0)
        p = np.random.normal(0, 0.01)*np.matrix(e).T*np.sqrt(np.matrix(l)).T
        for c in range(3):
            numpy_image[c] += p[c]
        numpy_image = np.clip(numpy_image, 0, 1)
        # return augmented data.
        return torch.Tensor(numpy_image)


class Scale(object):
    """ Divide the input pose by the given value.

    Args:
        value (int): Divide value.
    """

    def __init__(self, value=227):
        self.value = value

    def __call__(self, pose):
        return pose/self.value
