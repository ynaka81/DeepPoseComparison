# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_, ok_
from mock import patch
import numpy as np
from PIL import Image

from modules.dataset_indexing.chainer import PoseDataset


class TestPoseDatasetConstructure(unittest.TestCase):

    @patch('modules.dataset_indexing.chainer.pose_dataset.open')
    def test_load_dataset(self, mock):
        # prepare mock.
        mock.return_value = ['image1.png,1.0,2.0,0.0,3.0,4.0,1.0\n',
                             'image2.png,5.0,6.0,1.0,7.0,8.0,0.0\n']
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

    @patch('modules.dataset_indexing.chainer.pose_dataset.open')
    def setUp(self, mock):
        # prepare mock.
        mock.return_value = ['image1.png,1.0,2.0,0.0,3.0,4.0,1.0,5.0,6.0,0.0\n',
                             'image2.png,7.0,8.0,1.0,9.0,8.0,0.0,7.0,6.0,1.0\n']
        # set up.
        self.path = 'test_data'
        self.dataset = PoseDataset(self.path)

    @patch('PIL.Image.open', return_value=Image.new('RGB', (55, 75)))
    def test_read_image(self, mock):
        image = self.dataset._read_image('dummy.png')
        eq_(image.dtype, np.float32)
        eq_(image.shape, (3, 75, 55))

    def test_crop_image(self):
        self.dataset.data_augmentation = False
        image = np.arange(256*256*3).reshape((3, 256, 256))
        # crop on a pose center
        pose = np.array([[108, 50], [148, 180]])
        cropped_image, moved_pose = self.dataset._crop_image(image, pose)
        eq_(cropped_image.shape, (3, 227, 227))
        ok_((cropped_image == image[:, 1:228, 14:241]).all())
        correct = np.array([[94, 49], [134, 179]])
        ok_((moved_pose == correct).all())
        # left side is too tight
        pose = np.array([[40, 50], [160, 180]])
        cropped_image, moved_pose = self.dataset._crop_image(image, pose)
        eq_(cropped_image.shape, (3, 227, 227))
        ok_((cropped_image == image[:, 1:228, :227]).all())
        correct = np.array([[40, 49], [160, 179]])
        ok_((moved_pose == correct).all())
        # right side is too tight
        pose = np.array([[100, 50], [200, 180]])
        cropped_image, moved_pose = self.dataset._crop_image(image, pose)
        eq_(cropped_image.shape, (3, 227, 227))
        ok_((cropped_image == image[:, 1:228, 29:]).all())
        correct = np.array([[71, 49], [171, 179]])
        ok_((moved_pose == correct).all())

    def test_crop_image_data_augmentation(self):
        self.dataset.data_augmentation = False
        image = np.zeros((3, 256, 256))
        pose = np.zeros((2, 2))
        for i in range(20):
            cropped_image, _ = self.dataset._crop_image(image, pose)
            eq_(cropped_image.shape, (3, 227, 227))
            # p_min = np.min(pose_i, 0)
            # p_max = np.max(pose_i, 0)
            # ok_((p_min >= 1).all())
            # ok_((p_max < shape - 1).all())

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
        shape = (256, 256)
        mock.side_effect = [Image.new('RGB', shape),
                            Image.new('RGB', shape),
                            Image.new('RGB', shape),
                            Image.new('RGB', shape)]
        # test.
        for flag in (True, False):
            self.dataset.data_augmentation = flag
            for i in range(len(self.dataset)):
                image, pose, visibility = self.dataset.get_example(i)
                # test for image.
                eq_(image.shape, (3, 227, 227))
                self.assertGreaterEqual(np.min(image), 0)
                self.assertLessEqual(np.max(image), 1)
                # # test for pose.
                # p_min = np.min(pose, 0)
                # p_max = np.max(pose, 0)
                # ok_((p_min >= 1).all())
                # ok_((p_max < shape - 1).all())
                # test for visibility.
                eq_(visibility.shape, (3, 1))
