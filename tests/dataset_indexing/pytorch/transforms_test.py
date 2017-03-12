# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_, ok_
import numpy as np
import torch

from modules.dataset_indexing.pytorch import Crop, RandomNoise, Scale


class TestCrop(unittest.TestCase):

    def setUp(self):
        self.transform = Crop()

    def test_call(self):
        self.transform.data_augmentation = False
        image = torch.range(0, 256*256*3 - 1).view(3, 256, 256)
        visibility = np.ones((2, 1), dtype=np.int32)
        # crop on a pose center
        pose = np.array([[108, 50], [148, 180]], dtype=np.float32)
        transformed_image, transformed_pose, transformed_visibility = self.transform(image, pose, visibility)
        eq_(transformed_image.size(), (3, 227, 227))
        ok_((transformed_image == image[:, 1:228, 14:241]).all())
        eq_(transformed_pose.dtype, np.float32)
        correct = np.array([[94, 49], [134, 179]])
        ok_((transformed_pose == correct).all())
        ok_((transformed_visibility == visibility).all())
        # left side is too tight
        pose = np.array([[40, 50], [160, 180]], dtype=np.float32)
        transformed_image, transformed_pose, transformed_visibility = self.transform(image, pose, visibility)
        eq_(transformed_image.size(), (3, 227, 227))
        ok_((transformed_image == image[:, 1:228, :227]).all())
        eq_(transformed_pose.dtype, np.float32)
        correct = np.array([[40, 49], [160, 179]])
        ok_((transformed_pose == correct).all())
        ok_((transformed_visibility == visibility).all())
        # right side is too tight
        pose = np.array([[100, 50], [200, 180]], dtype=np.float32)
        transformed_image, transformed_pose, transformed_visibility = self.transform(image, pose, visibility)
        eq_(transformed_image.size(), (3, 227, 227))
        ok_((transformed_image == image[:, 1:228, 29:]).all())
        eq_(transformed_pose.dtype, np.float32)
        correct = np.array([[71, 49], [171, 179]])
        ok_((transformed_pose == correct).all())
        ok_((transformed_visibility == visibility).all())

    def test_call_data_augmentation(self):
        self.transform.data_augmentation = True
        image = torch.zeros(3, 256, 256)
        visibility = np.ones((2, 1), dtype=np.int32)
        for i in range(20):
            pose = np.random.rand(2, 2).astype(np.float32)*227
            transformed_image, transformed_pose, transformed_visibility = self.transform(image, pose, visibility)
            eq_(transformed_image.size(), (3, 227, 227))
            eq_(transformed_pose.dtype, np.float32)
            ok_((transformed_pose >= 0).all())
            ok_((transformed_pose <= 227).all())
            ok_((transformed_visibility == visibility).all())

class TestRandomNoise(unittest.TestCase):

    def setUp(self):
        self.transform = RandomNoise()

    def _calculate_image_eigen(self, image):
        C = np.cov(np.reshape(image, (3, -1)))
        l, e = np.linalg.eig(C)
        return l

    def test_call(self):
        image = np.random.rand(3, 256, 256).astype(np.float32)
        l = self._calculate_image_eigen(image)
        image = torch.Tensor(image)
        diff = []
        for i in range(100):
            noise_image = self.transform(image)
            eq_(type(image), torch.FloatTensor)
            eq_(image.size(), (3, 256, 256))
            ok_((noise_image >= 0).all())
            ok_((noise_image <= 1).all())
            l_noise = self._calculate_image_eigen(noise_image.numpy())
            diff.append(np.linalg.norm(l - l_noise))
        self.assertAlmostEqual(np.mean(diff), 0, delta=0.1*np.linalg.norm(l))

class TestScale(unittest.TestCase):

    def setUp(self):
        self.value = 227
        self.transform = Scale(self.value)

    def test_call(self):
        for i in range(100):
            pose = np.random.rand(14, 2).astype(np.float32)*self.value
            transformed_pose = self.transform(pose)
            eq_(transformed_pose.dtype, np.float32)
            ok_((transformed_pose >= 0).all())
            ok_((transformed_pose <= 1).all())
