# -*- coding: utf-8 -*-
# pylint: skip-file

import os
import unittest
from nose.tools import eq_, ok_, raises
from mock import patch
import numpy as np

from modules.errors import FileNotFoundError, SaveImageFailed, CropFailed
from modules.datasets import LSPDatasetGenerator


class TestLSPDatasetGeneratorConstructure(unittest.TestCase):

    def _test_init(self, mock):
        output = 'test_data'
        LSPDatasetGenerator(output=output)
        mock.assert_called_once_with(os.path.join(output, 'images'))

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
        self.output = 'test_data'
        self.generator = LSPDatasetGenerator(path=self.path, output=self.output)

    def tearDown(self):
        pass

    @patch('modules.datasets.generator.loadmat', return_value={'joints':np.zeros((3, 14, 10))})
    def test_load_joints_lsp(self, mock):
        dataset_name = 'lsp_dataset'
        joints = self.generator._load_joints(dataset_name)
        eq_(joints.shape, (10, 14, 3))
        mock.assert_called_once_with(os.path.join(self.path, dataset_name, 'joints.mat'))

    @patch('modules.datasets.generator.loadmat', return_value={'joints':np.zeros((14, 3, 10))})
    def test_load_joints_lspet(self, mock):
        dataset_name = 'lspet_dataset'
        joints = self.generator._load_joints(dataset_name)
        eq_(joints.shape, (10, 14, 3))
        mock.assert_called_once_with(os.path.join(self.path, dataset_name, 'joints.mat'))

    @patch('cv2.imread', return_value=np.zeros((320, 240)))
    def test_load_image_lsp(self, mock):
        dataset_name = 'lsp_dataset'
        image_file, image = self.generator._load_image(dataset_name, 1)
        eq_(image_file, 'im0001.jpg')
        eq_(image.shape, (320, 240))
        mock.assert_called_once_with(os.path.join(self.path, dataset_name, 'images', 'im0001.jpg'))

    @patch('cv2.imread', return_value=np.zeros((320, 240)))
    def test_load_image_lspet(self, mock):
        dataset_name = 'lspet_dataset'
        image_file, image = self.generator._load_image(dataset_name, 1)
        eq_(image_file, 'im00001.jpg')
        eq_(image.shape, (320, 240))
        mock.assert_called_once_with(os.path.join(self.path, dataset_name, 'images', 'im00001.jpg'))

    @raises(FileNotFoundError)
    @patch('cv2.imread', return_value=None)
    def test_load_image_raise_error(self, mock):
        dataset_name = 'test_dataset'
        image_file, image = self.generator._load_image(dataset_name, 1)

    def test_validate_crop_feasibility(self):
        crop = self.generator._validate_crop_feasibility(11, 5, 6)
        eq_(crop, 0)
        crop = self.generator._validate_crop_feasibility(13, 5, 6)
        eq_(crop, 2)

    @raises(CropFailed)
    def test_validate_crop_feasibility_too_tight_originally_min(self):
        self.generator._validate_crop_feasibility(13, 0, 10)

    @raises(CropFailed)
    def test_validate_crop_feasibility_too_tight_originally_max(self):
        self.generator._validate_crop_feasibility(11, 1, 10)

    @raises(CropFailed)
    def test_validate_crop_feasibility_too_tight(self):
        self.generator._validate_crop_feasibility(13, 1, 10)

    def test_crop_image(self):
        W, H = 13, 14
        image = np.arange(H*W*3).reshape((H, W, 3))
        # crop left&top side
        joint = np.array([[3, 4, 0], [10, 10, 0]])
        cropped = self.generator._crop_image(image, joint)
        h, w, _ = cropped.shape
        eq_((w - self.generator.ksize)%self.generator.stride, 0)
        eq_((h - self.generator.ksize)%self.generator.stride, 0)
        ok_((cropped == image[3:, 2:, :]).all())
        # crop right&bottom side
        joint = np.array([[3, 4, 0], [9, 9, 0]])
        cropped = self.generator._crop_image(image, joint)
        h, w, _ = cropped.shape
        eq_((w - self.generator.ksize)%self.generator.stride, 0)
        eq_((h - self.generator.ksize)%self.generator.stride, 0)
        ok_((cropped == image[:11, :11, :]).all())
        # crop left&bottom side
        joint = np.array([[3, 4, 0], [10, 9, 0]])
        cropped = self.generator._crop_image(image, joint)
        h, w, _ = cropped.shape
        eq_((w - self.generator.ksize)%self.generator.stride, 0)
        eq_((h - self.generator.ksize)%self.generator.stride, 0)
        ok_((cropped == image[:11, 2:, :]).all())

    def test_crop_image_tight(self):
        W, H = 13, 14
        image = np.arange(H*W*3).reshape((H, W, 3))
        # crop left&top side
        joint = np.array([[2, 3, 0], [10, 11, 0]])
        cropped = self.generator._crop_image(image, joint)
        h, w, _ = cropped.shape
        eq_((w - self.generator.ksize)%self.generator.stride, 0)
        eq_((h - self.generator.ksize)%self.generator.stride, 0)
        ok_((cropped == image[2:13, 1:12, :]).all())
        # crop right&bottom side
        joint = np.array([[2, 2, 0], [10, 10, 0]])
        cropped = self.generator._crop_image(image, joint)
        h, w, _ = cropped.shape
        eq_((w - self.generator.ksize)%self.generator.stride, 0)
        eq_((h - self.generator.ksize)%self.generator.stride, 0)
        ok_((cropped == image[1:12, 1:12, :]).all())

    @patch('cv2.imwrite', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_save_image(self, mock_isdir, mock_imwrite):
        dataset_name = 'lsp_dataset'
        image_file = 'im0001.jpg'
        image = np.zeros((320, 240))
        image_path = self.generator._save_image(dataset_name, image_file, image)
        path = os.path.join(self.output, 'images', dataset_name)
        eq_(image_path, os.path.join(path, image_file))
        mock_isdir.assert_called_once_with(path)
        mock_imwrite.assert_called_once_with(image_path, image)

    @patch('cv2.imwrite', return_value=True)
    @patch('os.makedirs')
    @patch('os.path.isdir', return_value=False)
    def test_save_image_directory_not_exist(self, mock_isdir, mock_makedirs, mock_imwrite):
        dataset_name = 'lsp_dataset'
        image_file = 'im0001.jpg'
        image = np.zeros((320, 240))
        image_path = self.generator._save_image(dataset_name, image_file, image)
        path = os.path.join(self.output, 'images', dataset_name)
        eq_(image_path, os.path.join(path, image_file))
        mock_makedirs.assert_called_once_with(path)

    @raises(SaveImageFailed)
    @patch('cv2.imwrite', return_value=False)
    @patch('os.path.isdir', return_value=True)
    def test_save_image_raise_error(self, mock_isdir, mock_imwrite):
        dataset_name = 'lsp_dataset'
        image_file = 'im0001.jpg'
        image = np.zeros((320, 240))
        self.generator._save_image(dataset_name, image_file, image)

    def test_get_data_label_lsp_train(self):
        label = self.generator._get_data_label('lsp_dataset', 1000)
        eq_(label, 'train')
        label = self.generator._get_data_label('lsp_dataset', 1001)
        eq_(label, 'test')
        label = self.generator._get_data_label('lspet_dataset', 1)
        eq_(label, 'train')

    def test_make_dataset_line(self):
        image_path = 'test_path'
        joint = np.zeros((14, 3))
        line = self.generator._make_dataset_line(image_path, joint)
        eq_(line, image_path + ',' + ','.join(['0.0']*14*3) + os.linesep)

    @patch('cv2.imwrite', return_value=True)
    @patch('os.path.isdir', return_value=True)
    @patch('cv2.imread', return_value=np.zeros((14, 13, 3)))
    @patch('modules.datasets.generator.loadmat')
    def test_generate_datasets(self, m1, m2, m3, m4):
        # prepare mock.
        joints = np.array([[[1, 2, 0], [3, 4, 0]],
                           [[5, 6, 0], [7, 8, 0]],
                           [[0, 0, 0], [1, 1, 0]]])
        m1.side_effect = [{'joints': joints.transpose(2, 1, 0)},
                          {'joints': joints.transpose(1, 2, 0)}]
        # test case.
        datasets = self.generator._generate_datasets()
        eq_(datasets['test'], [])
        train = ['{0},1,2,0,3,4,0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0001.jpg')),
                 '{0},5,6,0,7,8,0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0002.jpg')),
                 '{0},1,2,0,3,4,0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00001.jpg')),
                 '{0},5,6,0,7,8,0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00002.jpg'))]
        eq_(datasets['train'], train)

    @patch('modules.datasets.generator.open')
    def test_write_datasets(self, mock):
        datasets = {'test1': ['a,1,2,3,4,5,6', 'b,7,8,9,1,2,3'], 'test2': ['c,4,5,6,7,8,9']}
        self.generator._write_datasets(datasets)
        eq_(mock.call_args_list,
            [((os.path.join(self.output, 'test1'), 'w'),),
             ((os.path.join(self.output, 'test2'), 'w'),)])
        eq_(mock.return_value.write.call_args_list,
            [(('a,1,2,3,4,5,6',),),
             (('b,7,8,9,1,2,3',),),
             (('c,4,5,6,7,8,9',),)])

    @patch('modules.datasets.generator.open')
    @patch('cv2.imwrite', return_value=True)
    @patch('os.path.isdir', return_value=True)
    @patch('cv2.imread', return_value=np.zeros((14, 13, 3)))
    @patch('modules.datasets.generator.loadmat')
    def test_generate(self, m1, m2, m3, m4, m5):
        # prepare mock.
        joints = np.array([[[1, 2, 0], [3, 4, 0]],
                           [[5, 6, 0], [12, 13, 0]],
                           [[5, 6, 0], [7, 8, 0]]])
        m1.side_effect = [{'joints': joints.transpose(2, 1, 0)},
                          {'joints': joints.transpose(1, 2, 0)}]
        # test case.
        self.generator.generate()
        eq_(m5.call_args_list,
            [((os.path.join(self.output, 'test'), 'w'),),
             ((os.path.join(self.output, 'train'), 'w'),)])
        train = [(('{0},1,2,0,3,4,0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0001.jpg')),),),
                 (('{0},5,6,0,7,8,0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0003.jpg')),),),
                 (('{0},1,2,0,3,4,0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00001.jpg')),),),
                 (('{0},5,6,0,7,8,0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00003.jpg')),),)]
        eq_(m5.return_value.write.call_args_list, train)
