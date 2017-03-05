# -*- coding: utf-8 -*-
""" Pose dataset indexing. """

import random
import numpy as np
from PIL import Image
from chainer import dataset


class PoseDataset(dataset.DatasetMixin):
    """ Pose dataset indexing.

    Args:
        path (str): A path to dataset.
        data_augmentation (bool): True for data augmentation.
        stride (int): Stride of filter applications.
    """

    def __init__(self, path, data_augmentation=True, stride=4):
        self.path = path
        self.data_augmentation = data_augmentation
        self.stride = stride
        # load dataset.
        self.images, self.poses = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """ Returns the i-th example. """
        image = self._read_image(self.images[i])
        pose = self.poses[i]
        # data augumentation.
        if self.data_augmentation:
            image, pose = self._random_crop(image, pose)
            image = self._random_noise(image)
        # scale to [0, 1].
        image /= 255.
        return image, pose

    def _load_dataset(self):
        images = []
        poses = []
        for line in open(self.path):
            line_split = line[:-1].split(',')
            images.append(line_split[0])
            x = map(int, line_split[1:])
            x = np.matrix(x).reshape(-1, 3)
            poses.append(x)
        return images, poses

    @staticmethod
    def _read_image(path):
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=np.float32)
        finally:
            f.close()
        return image.transpose(2, 0, 1)

    def _random_crop(self, image, pose):
        p_min = np.min(pose, 0)
        p_max = np.max(pose, 0)
        h, w, _ = image.shape
        shape = (w, h)
        crop_min = [0, 0]
        crop_max = [0, 0]
        # crop image.
        for i in range(2):
            residual = shape - (np.ceil(p_max[i] - p_min[i]) + 3)
            crop_residual = residual/self.stride*self.stride
            random_all = random.randint(0, crop_residual)
            crop_min[i] = random.randint(0, p_min[i])
            crop_max[i] = random_all - crop_min[i]
        image = image[:, crop_min[1]:crop_max[1], crop_min[0]:crop_max[0]]
        # modify pose according to the cropping.
        pose = pose - np.array(crop_min + [0])
        # return augmented data.
        return image, pose

    @staticmethod
    def _random_noise(image):
        image = image.copy()
        # add random noise to keep eigen value.
        C = np.cov(np.reshape(image, (3, -1)))
        l, e = np.linalg.eig(C)
        p = np.random.normal(0, 0.1)*np.matrix(e).T*np.sqrt(np.matrix(l)).T
        for c in range(3):
            image[c] += p[c]
        image = np.clip(image, 0, 255)
        # return augmented data.
        return image
