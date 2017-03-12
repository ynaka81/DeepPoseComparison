# -*- coding: utf-8 -*-
""" Transform input and outpupt data. """

import random
import numpy as np


# pylint: disable=too-few-public-methods
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
        _, height, width = image.shape
        shape = (width, height)
        visible_pose = pose[visibility.ravel().astype(bool)]
        p_min = np.min(visible_pose, 0)
        p_max = np.max(visible_pose, 0)
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
        cropped_image = image[:, crop_shape[2]:crop_shape[3], crop_shape[0]:crop_shape[1]]
        moved_pose = pose - np.array((crop_shape[0], crop_shape[2]), dtype=np.float32)
        return cropped_image, moved_pose

# pylint: disable=too-few-public-methods
class RandomNoise(object):
    """ Give random noise to the given PIL.Image.
    """

    def __call__(self, image):
        image = image.copy()
        # add random noise to keep eigen value.
        C = np.cov(np.reshape(image, (3, -1)))
        l, e = np.linalg.eig(C)
        l = np.maximum(l, 0)
        p = np.random.normal(0, 0.1)*np.matrix(e).T*np.sqrt(np.matrix(l)).T
        for c in range(3):
            image[c] += p[c]
        image = np.clip(image, 0, 1)
        # return augmented data.
        return image

# pylint: disable=too-few-public-methods
class Scale(object):
    """ Divide the input pose by the given value.

    Args:
        value (int): Divide value.
    """

    def __init__(self, value=227):
        self.value = value

    def __call__(self, pose):
        return pose/self.value
