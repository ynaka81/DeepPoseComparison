# -*- coding: utf-8 -*-

import os
import unittest
from nose.tools import eq_, ok_
from mock import patch
import numpy as np

from modules.datasets.lspet.dataset import LSPETDataset


class TestLSPETDataset(unittest.TestCase):

    @patch('os.makedirs')
    def setUp(self, mock):
        self.path = 'test_orig_data'
        self.dataset = LSPETDataset(self.path)

    def test_get_extract_path(self):
        extract_path = self.dataset._get_extract_path()
        eq_(extract_path, os.path.join(self.path, 'lspet_dataset'))

    @patch('modules.datasets.lspet.dataset.loadmat', return_value={'joints': np.zeros((14, 3, 10))})
    def test_load_joints(self, mock):
        joints = self.dataset._load_joints()
        eq_(joints.shape, (10, 14, 3))
        correct = np.zeros((10, 14, 3))
        ok_((joints == correct).all())

    @patch('cv2.imread', return_value=np.zeros((320, 240)))
    def test_get_image(self, mock):
        image_file, image = self.dataset._get_image(0)
        eq_(image_file, 'im00001.jpg')
        eq_(image.shape, (320, 240))

    def test_get_data_label(self):
        eq_(self.dataset._get_data_label(0), 'train')
        eq_(self.dataset._get_data_label(1000), 'train')
