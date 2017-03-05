# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_, ok_
from mock import patch
import numpy as np
from PIL import Image

from modules.dataset_indexing import PoseDataset


class TestPoseDatasetConstructure(unittest.TestCase):

    @patch('modules.dataset_indexing.pose_dataset.open')
    def test_load_dataset(self, mock):
        # prepare mock.
        mock.return_value = ['image1.png,1,2,0,3,4,1\n',
                             'image2.png,5,6,1,7,8,0\n']
        # test.
        dataset = PoseDataset('test_data')
        eq_(dataset.images, ['image1.png', 'image2.png'])
        correct = [np.array([[1, 2],
                              [3, 4]], dtype=np.float32),
                   np.array([[5, 6],
                              [7, 8]], dtype=np.float32)]
        for p, c in zip(dataset.poses, correct):
            ok_((p == c).all())
        correct = [np.array([[0], [1]], dtype=np.int32),
                   np.array([[1], [0]], dtype = np.int32)]
        for v, c in zip(dataset.visibilities, correct):
            ok_((v == c).all())

class TestPoseDataset(unittest.TestCase):

    @patch('modules.dataset_indexing.pose_dataset.open')
    def setUp(self, mock):
        # prepare mock.
        mock.return_value = ['image1.png,1,2,0,3,4,1,5,6,0\n',
                             'image2.png,7,8,1,9,8,0,7,6,1\n']
        # set up.
        self.path = 'test_data'
        self.ksize = 11
        self.stride = 4
        self.dataset = PoseDataset(self.path, ksize=self.ksize, stride=self.stride)

    @patch('PIL.Image.open', return_value=Image.new('RGB', (55, 75)))
    def test_read_image(self, mock):
        image = self.dataset._read_image('dummy.png')
        eq_(image.dtype, np.float32)
        eq_(image.shape, (3, 75, 55))

    def test_random_crop(self):
        image = np.zeros((3, 19, 15))
        pose = np.array([[3, 5],
                         [9, 14]])
        for i in range(20):
            image_i, pose_i = self.dataset._random_crop(image, pose)
            _, h, w = image_i.shape
            shape = np.array((w, h))
            ok_((((shape - self.ksize)%self.stride) == 0).all())
            p_min = np.min(pose_i, 0)
            p_max = np.max(pose_i, 0)
            ok_((p_min >= 1).all())
            ok_((p_max < shape - 1).all())

    def _calculate_image_eigen(self, image):
        C = np.cov(np.reshape(image, (3, -1)))
        l, e = np.linalg.eig(C)
        return l

    def test_random_noise(self):
        image = np.random.randint(0, 256, (3, 15, 11))
        image = np.array(image, dtype=np.float32)
        l = self._calculate_image_eigen(image)
        diff = []
        for i in range(100):
            noise_image = self.dataset._random_noise(image)
            self.assertGreaterEqual(np.min(noise_image), 0)
            self.assertLessEqual(np.max(noise_image), 255)
            l_noise = self._calculate_image_eigen(noise_image)
            diff.append(np.linalg.norm(l - l_noise))
        self.assertAlmostEqual(np.mean(diff), 0, delta=0.1*np.linalg.norm(l))

    def test_len(self):
        eq_(len(self.dataset), 2)

    @patch('PIL.Image.open')
    def test_get_example(self, mock):
        # prepare mock.
        mock.side_effect = [Image.new('RGB', (55, 75)),
                            Image.new('RGB', (11, 11)),
                            Image.new('RGB', (55, 75)),
                            Image.new('RGB', (55, 75))]
        # test.
        for flag in (True, False):
            self.dataset.data_augmentation = flag
            for i in range(len(self.dataset)):
                image, pose, visibility = self.dataset.get_example(i)
                # test for image.
                _, h, w = image.shape
                shape = np.array((h, w))
                ok_((((shape - self.ksize)%self.stride) == 0).all())
                self.assertGreaterEqual(np.min(image), 0)
                self.assertLessEqual(np.max(image), 1)
                if not flag:
                    eq_(image.shape, (3, 75, 55))
                # test for pose.
                p_min = np.min(pose, 0)
                p_max = np.max(pose, 0)
                ok_((p_min >= 1).all())
                ok_((p_max < shape - 1).all())
                # test for visibility.
                eq_(visibility.shape, (3, 1))
