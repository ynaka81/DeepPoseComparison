# -*- coding: utf-8 -*-
# pylint: skip-file

import os
import unittest
from nose.tools import eq_, ok_, raises
from mock import patch
import numpy as np

from modules.errors import FileNotFoundError, SaveImageFailed
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
        correct = np.zeros((10, 14, 3))
        correct[:, :, 2] = 1
        ok_((joints == correct).all())
        mock.assert_called_once_with(os.path.join(self.path, dataset_name, 'joints.mat'))

    @patch('modules.datasets.generator.loadmat', return_value={'joints':np.zeros((14, 3, 10))})
    def test_load_joints_lspet(self, mock):
        dataset_name = 'lspet_dataset'
        joints = self.generator._load_joints(dataset_name)
        eq_(joints.shape, (10, 14, 3))
        ok_((joints == np.zeros((10, 14, 3))).all())
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

    def test_scale_image(self):
        joint = np.array([[20, 30, 0], [40, 50, 0]])
        # scale up
        image = np.zeros((160, 120, 3))
        scaled_image, scaled_joint = self.generator._scale_image(image, joint)
        eq_(scaled_image.shape[1], 256)
        correct = joint*2.13
        ok_((scaled_joint - correct < 1.).all())
        # scale down
        image = np.zeros((480, 620, 3))
        scaled_image, scaled_joint = self.generator._scale_image(image, joint)
        eq_(scaled_image.shape[0], 256)
        correct = joint*0.53
        ok_((scaled_joint - correct < 1.).all())

    def test_crop_image(self):
        # crop width side
        W, H = 320, 256
        image = np.arange(H*W*3).reshape((H, W, 3))
        # crop on a joint center
        joint = np.array([[100, 30, 0], [160, 50, 0]])
        cropped_image, cropped_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[:, 2:258, :]).all())
        correct = np.array([[98, 30, 0], [158, 50, 0]])
        ok_((cropped_joint == correct).all())
        # left side is too tight
        joint = np.array([[20, 30, 0], [40, 50, 0]])
        cropped_image, cropped_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[:, :256, :]).all())
        correct = np.array([[20, 30, 0], [40, 50, 0]])
        ok_((cropped_joint == correct).all())
        # left side is too tight
        joint = np.array([[200, 30, 0], [400, 50, 0]])
        cropped_image, cropped_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[:, 64:, :]).all())
        correct = np.array([[136, 30, 0], [336, 50, 0]])
        ok_((cropped_joint == correct).all())
        # crop height side
        W, H = 256, 320
        image = np.arange(H*W*3).reshape((H, W, 3))
        # crop on a joint center
        joint = np.array([[20, 140, 0], [40, 240, 0]])
        cropped_image, cropped_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[62:318, :, :]).all())
        correct = np.array([[20, 78, 0], [40, 178, 0]])
        ok_((cropped_joint == correct).all())

    def test_validate(self):
        joint = np.array([[20, 30, 1], [40, 50, 0]])
        eq_(self.generator._validate(joint), True)
        joint = np.array([[0, 30, 1], [40, 50, 0]])
        eq_(self.generator._validate(joint), False)
        joint = np.array([[20, 30, 1], [40, 256, 0]])
        eq_(self.generator._validate(joint), False)
        joint = np.array([[10, 30, 1], [237, 40, 0]])
        eq_(self.generator._validate(joint), False)

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
    @patch('cv2.imread', return_value=np.zeros((128, 192, 3)))
    @patch('modules.datasets.generator.loadmat')
    def test_generate_datasets(self, m1, m2, m3, m4):
        # prepare mock.
        joints = np.array([[[50, 60, 0], [150, 100, 1]],
                           [[100, 40, 1], [120, 80, 0]],
                           [[40, 0, 0], [180, 1, 0]]])
        m1.side_effect = [{'joints': joints.transpose(2, 1, 0).copy()},
                          {'joints': joints.transpose(1, 2, 0).copy()}]
        # test case.
        datasets = self.generator._generate_datasets()
        eq_(datasets['test'], [])
        train = ['{0},28.0,120.0,1.0,228.0,200.0,0.0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0001.jpg')),
                 '{0},108.0,80.0,0.0,148.0,160.0,1.0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0002.jpg')),
                 '{0},28.0,120.0,0.0,228.0,200.0,1.0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00001.jpg')),
                 '{0},108.0,80.0,1.0,148.0,160.0,0.0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00002.jpg'))]
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
    @patch('cv2.imread', return_value=np.zeros((128, 192, 3)))
    @patch('modules.datasets.generator.loadmat')
    def test_generate(self, m1, m2, m3, m4, m5):
        # prepare mock.
        joints = np.array([[[50, 60, 0], [150, 100, 1]],
                           [[40, 0, 0], [180, 1, 0]],
                           [[100, 40, 1], [120, 80, 0]]])
        m1.side_effect = [{'joints': joints.transpose(2, 1, 0).copy()},
                          {'joints': joints.transpose(1, 2, 0).copy()}]
        # test case.
        self.generator.generate()
        eq_(m5.call_args_list,
            [((os.path.join(self.output, 'test'), 'w'),),
             ((os.path.join(self.output, 'train'), 'w'),)])
        train = [(('{0},28.0,120.0,1.0,228.0,200.0,0.0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0001.jpg')),),),
                 (('{0},108.0,80.0,0.0,148.0,160.0,1.0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0003.jpg')),),),
                 (('{0},28.0,120.0,0.0,228.0,200.0,1.0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00001.jpg')),),),
                 (('{0},108.0,80.0,1.0,148.0,160.0,0.0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00003.jpg')),),)]
        eq_(m5.return_value.write.call_args_list, train)
