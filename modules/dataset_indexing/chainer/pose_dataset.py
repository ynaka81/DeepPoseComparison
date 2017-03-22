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
        crop_size (int): Size of cropping for DNN training.
    """

    def __init__(self, path, data_augmentation=True, crop_size=227):
        self.path = path
        self.data_augmentation = data_augmentation
        self.crop_size = crop_size
        # load dataset.
        self.images, self.poses, self.visibilities = self._load_dataset()

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """ Returns the i-th example. """
        image = self._read_image(self.images[i])
        pose = self.poses[i]
        visibility = self.visibilities[i]
        # crop image.
        image, pose = self._crop_image(image, pose, visibility)
        # add random noise for data augumentation.
        if self.data_augmentation:
            image = self._random_noise(image)
        # scale to [0, 1].
        image /= 255.
        pose /= self.crop_size
        return image, pose, visibility

    def _load_dataset(self):
        images = []
        poses = []
        visibilities = []
        for line in open(self.path):
            line_split = line[:-1].split(',')
            images.append(line_split[0])
            x = np.array(line_split[1:])
            x = x.reshape(-1, 3)
            poses.append(x[:, :2].astype(np.float32))
            visibilities.append(x[:, 2].reshape(-1, 1).astype(np.float32).astype(np.int32))
        return images, poses, visibilities

    @staticmethod
    def _read_image(path):
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=np.float32)
        finally:
            f.close()
        return image.transpose(2, 0, 1)

    def _crop_image(self, image, pose, visibility):
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

    @staticmethod
    def _random_noise(image):
        image = image.copy()
        # add random noise to keep eigen value.
        C = np.cov(np.reshape(image, (3, -1)))
        C = (C + C.T)/2
        l, e = np.linalg.eig(C)
        l = np.maximum(l, 0)
        p = np.random.normal(0, 0.01)*np.matrix(e).T*np.sqrt(np.matrix(l)).T
        for c in range(3):
            image[c] += p[c]
        image = np.clip(image, 0, 255)
        # return augmented data.
        return image
