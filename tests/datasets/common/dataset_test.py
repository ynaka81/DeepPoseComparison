# -*- coding: utf-8 -*-

import unittest
from nose.tools import eq_, raises
from mock import patch
import numpy as np

from modules.errors import FileNotFoundError
from modules.datasets.common.dataset import Dataset

DATASET_NAME = 'test_dataset'
URL = 'http://dummy/test_dataset.zip'
EXTRACT_PATH = 'test_extract_path'


class DatasetChild(Dataset):

    def __init__(self, path):
        super(DatasetChild, self).__init__(DATASET_NAME, URL, path)

    def _get_extract_path(self):
        return EXTRACT_PATH

    def _load_joints(self):
        return np.zeros((10, 14, 3))

    def _get_image(self, i):
        if i == 0:
            return 'im0001.jpg', np.zeros((320, 240))
        else:
            return None, None

    def _get_data_label(self, i):
        return 'test'


class TestDatasetConstructure(unittest.TestCase):

    def _test_init(self, mock):
        path = 'test_orig_data'
        DatasetChild(path)
        mock.assert_called_once_with(path)

    @patch('os.makedirs')
    def test_init_directory_not_exist(self, mock):
        self._test_init(mock)

    @patch('os.makedirs', side_effect=OSError)
    def test_init_directory_exist(self, mock):
        self._test_init(mock)


class TestDataset(unittest.TestCase):

    @patch('os.makedirs')
    def setUp(self, mock):
        self.dataset = DatasetChild('test_orig_data')

    @patch('os.path.isdir', return_value=True)
    def test_load(self, mock):
        self.dataset.load()
        eq_(self.dataset.joints.shape, (10, 14, 3))

    @patch('os.path.isdir', return_value=True)
    def test_get_data(self, mock):
        self.dataset.load()
        label, joint, image_file, image = self.dataset.get_data(0)
        eq_(label, 'test')
        eq_(joint.shape, (14, 3))
        eq_(image_file, 'im0001.jpg')
        eq_(image.shape, (320, 240))

    @raises(FileNotFoundError)
    @patch('os.path.isdir', return_value=True)
    def test_get_data_no_image(self, mock):
        self.dataset.load()
        self.dataset.get_data(1)

    @patch('os.path.isdir', return_value=True)
    def test_len(self, mock):
        self.dataset.load()
        eq_(len(self.dataset), 10)

    @patch('os.remove')
    @patch('zipfile.ZipFile')
    @patch('wget.download')
    @patch('os.path.isdir', return_value=False)
    def test_download(self, m1, m2, m3, m4):
        self.dataset._download()
        eq_(m1.call_count, 1)
        eq_(m2.call_count, 1)
        eq_(m3.call_count, 1)
        eq_(m4.call_count, 1)

    @patch('os.remove')
    @patch('zipfile.ZipFile')
    @patch('wget.download')
    @patch('os.path.isdir', return_value=True)
    def test_download_file_exist(self, m1, m2, m3, m4):
        self.dataset._download()
        eq_(m1.call_count, 1)
        eq_(m2.call_count, 0)
        eq_(m3.call_count, 0)
        eq_(m4.call_count, 0)
