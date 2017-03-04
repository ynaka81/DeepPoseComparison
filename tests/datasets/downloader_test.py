# -*- coding: utf-8 -*-
# pylint: skip-file

import os
import unittest
from nose.tools import eq_, ok_
from mock import patch

from modules.datasets import LSPDatasetDownloader


class TestLSPDatasetDownloaderConstructure(unittest.TestCase):

    def _test_init(self, mock):
        path = 'test_orig_data'
        LSPDatasetDownloader(path)
        mock.assert_called_once_with(path)

    @patch('os.makedirs')
    def test_init_directory_not_exist(self, mock):
        self._test_init(mock)

    @patch('os.makedirs', side_effect=OSError)
    def test_init_directory_exist(self, mock):
        self._test_init(mock)

class TestLSPDatasetDownloader(unittest.TestCase):

    @patch('os.makedirs')
    def setUp(self, mock):
        self.path = 'test_orig_data'
        self.downloader = LSPDatasetDownloader(self.path)

    def tearDown(self):
        pass

    @patch('os.path.isdir', return_value=False)
    def test_need_download(self, mock):
        dataset_name = 'test_dataset'
        ok_(self.downloader._need_download(dataset_name))
        mock.assert_called_once_with(os.path.join(self.path, dataset_name))

    @patch('wget.download', return_value='test_path')
    def test_download_dataset(self, mock):
        url = 'test_url'
        path = self.downloader._download_dataset(url)
        mock.assert_called_once_with(url, self.path)
        eq_(path, mock.return_value)

    @patch('zipfile.ZipFile')
    def test_extract_dataset_lsp(self, mock):
        path = 'test_path'
        dataset_name = 'lsp_dataset'
        self.downloader._extract_dataset(dataset_name, path)
        mock.assert_called_once_with(path, 'r')
        mock_with = mock.return_value.__enter__.return_value
        mock_with.extractall.assert_called_once_with(self.path)

    @patch('zipfile.ZipFile')
    def test_extract_dataset_lspet(self, mock):
        path = 'test_path'
        dataset_name = 'lspet_dataset'
        self.downloader._extract_dataset(dataset_name, path)
        mock.assert_called_once_with(path, 'r')
        extract_path = os.path.join(self.path, dataset_name)
        mock_with = mock.return_value.__enter__.return_value
        mock_with.extractall.assert_called_once_with(extract_path)

    @patch('os.remove')
    def test_cleanup(self, mock):
        path = 'test_path'
        self.downloader._cleanup(path)
        mock.assert_called_once_with(path)

    @patch('os.remove')
    @patch('zipfile.ZipFile')
    @patch('wget.download')
    @patch('os.path.isdir', return_value=False)
    def test_download_file_not_exist(self, m1, m2, m3, m4):
        self.downloader.download()
        eq_(m1.call_count, 2)
        eq_(m2.call_count, 2)
        eq_(m3.call_count, 2)
        eq_(m4.call_count, 2)

    @patch('os.remove')
    @patch('zipfile.ZipFile')
    @patch('wget.download')
    @patch('os.path.isdir', return_value=True)
    def test_download_file_exist(self, m1, m2, m3, m4):
        self.downloader.download()
        eq_(m1.call_count, 2)
        eq_(m2.call_count, 0)
        eq_(m3.call_count, 0)
        eq_(m4.call_count, 0)
