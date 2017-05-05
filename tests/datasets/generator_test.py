# -*- coding: utf-8 -*-

import os
import unittest
from nose.tools import eq_, ok_, raises
from mock import patch
import numpy as np

from modules.errors import SaveImageFailed
from modules.datasets import DatasetGenerator


class TestDatasetGeneratorConstructure(unittest.TestCase):

    def _test_init(self, mock):
        output = 'test_data'
        DatasetGenerator(output=output)
        mock.assert_any_call(os.path.join(output, 'images'))

    @patch('os.makedirs')
    def test_init_directory_not_exist(self, mock):
        self._test_init(mock)

    @patch('os.makedirs', side_effect=OSError)
    def test_init_directory_exist(self, mock):
        self._test_init(mock)


class TestDatasetDownloader(unittest.TestCase):

    @patch('modules.datasets.generator.LSPETDataset')
    @patch('modules.datasets.generator.LSPDataset')
    @patch('os.makedirs')
    def setUp(self, m, m_lsp, m_lspet):
        # prepare mock.
        joints = np.array([[[50, 80, 0], [50, 80, 1], [150, 260, 1], [150, 260, 0]],
                           [[100, 200, 1], [100, 200, 0], [120, 280, 0], [120, 280, 1]],
                           [[40, 10, 0], [40, 10, 1], [120, 290, 1], [120, 290, 0]]])
        m_lsp_instance = m_lsp.return_value
        m_lsp_instance.name = 'lsp_dataset'
        m_lsp_instance.__len__.return_value = 2
        lsp_joints = joints.copy()
        lsp_joints[:, :, 2] = np.logical_not(joints[:, :, 2]).astype(int)
        m_lsp_instance.get_data = lambda i: ('train', lsp_joints[i], 'im{0:04d}.jpg'.format(i + 1), np.zeros((300, 200, 3)))
        m_lspet_instance = m_lspet.return_value
        m_lspet_instance.name = 'lspet_dataset'
        m_lspet_instance.__len__.return_value = 2
        lspet_joints = joints.copy()
        m_lspet_instance.get_data = lambda i: ('train', lspet_joints[i], 'im{0:05d}.jpg'.format(i + 1), np.zeros((300, 200, 3)))
        # initialize.
        self.path = 'test_orig_data'
        self.output = 'test_data'
        self.generator = DatasetGenerator(path=self.path, output=self.output)

    def test_pad_image(self):
        joint = np.array([[10, 20, 1], [30, 40, 1]])
        # pad width and height side
        image = np.zeros((100, 100, 3))
        padded_image, moved_joint = self.generator._pad_image(image, joint)
        eq_(padded_image.shape, (256, 256, 3))
        correct = np.array([[88, 98, 1], [108, 118, 1]])
        ok_((moved_joint == correct).all())
        # pad only one side
        image = np.zeros((300, 100, 3))
        padded_image, moved_joint = self.generator._pad_image(image, joint)
        eq_(padded_image.shape, (300, 256, 3))
        correct = np.array([[88, 20, 1], [108, 40, 1]])
        ok_((moved_joint == correct).all())
        # not pad
        image = np.zeros((300, 300, 3))
        padded_image, moved_joint = self.generator._pad_image(image, joint)
        eq_(padded_image.shape, (300, 300, 3))
        correct = np.array([[10, 20, 1], [30, 40, 1]])
        ok_((moved_joint == correct).all())

    def test_crop_image(self):
        # crop width side
        W, H = 320, 256
        image = np.arange(H*W*3).reshape((H, W, 3))
        # crop on a joint center
        joint = np.array([[100, 30, 1], [160, 50, 1]])
        cropped_image, moved_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[:, 2:258, :]).all())
        correct = np.array([[98, 30, 1], [158, 50, 1]])
        ok_((moved_joint == correct).all())
        # left side is too tight
        joint = np.array([[20, 30, 1], [40, 50, 1]])
        cropped_image, moved_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[:, :256, :]).all())
        correct = np.array([[20, 30, 1], [40, 50, 1]])
        ok_((moved_joint == correct).all())
        # right side is too tight
        joint = np.array([[200, 30, 1], [400, 50, 1]])
        cropped_image, moved_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[:, 64:, :]).all())
        correct = np.array([[136, 30, 1], [336, 50, 1]])
        ok_((moved_joint == correct).all())
        # crop height side
        W, H = 256, 320
        image = np.arange(H*W*3).reshape((H, W, 3))
        joint = np.array([[20, 140, 1], [40, 240, 1]])
        cropped_image, moved_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[62:318, :, :]).all())
        correct = np.array([[20, 78, 1], [40, 178, 1]])
        ok_((moved_joint == correct).all())
        # crop both side
        W, H = 320, 320
        image = np.arange(H*W*3).reshape((H, W, 3))
        joint = np.array([[100, 140, 1], [160, 240, 1]])
        cropped_image, moved_joint = self.generator._crop_image(image, joint)
        eq_(cropped_image.shape, (256, 256, 3))
        ok_((cropped_image == image[62:318, 2:258, :]).all())
        correct = np.array([[98, 78, 1], [158, 178, 1]])
        ok_((moved_joint == correct).all())

    def test_validate(self):
        joint = np.array([[20, 30, 1], [40, 50, 1]])
        eq_(self.generator._validate(joint), True)
        joint = np.array([[0, 30, 1], [40, 50, 1]])
        eq_(self.generator._validate(joint), False)
        joint = np.array([[20, 30, 1], [40, 256, 1]])
        eq_(self.generator._validate(joint), False)
        joint = np.array([[10, 30, 1], [237, 40, 1]])
        eq_(self.generator._validate(joint), False)
        joint = np.array([[20, 30, 1], [-40, -50, 0]])
        eq_(self.generator._validate(joint), True)
        joint = np.array([[20, 30, 0], [40, 50, 0]])
        eq_(self.generator._validate(joint), True)

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

    def test_make_dataset_line(self):
        image_path = 'test_path'
        joint = np.zeros((14, 3))
        line = self.generator._make_dataset_line(image_path, joint)
        eq_(line, image_path + ',' + ','.join(['0.0']*14*3) + os.linesep)

    @patch('cv2.imwrite', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_generate_datasets(self, m1, m2):
        datasets = self.generator._generate_datasets()
        eq_(datasets['test'], [])
        train = ['{0},78,38,1,78,38,0,178,218,0,178,218,1\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0001.jpg')),
                 '{0},128,156,0,128,156,1,148,236,1,148,236,0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0002.jpg')),
                 '{0},78,38,0,78,38,1,178,218,1,178,218,0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00001.jpg')),
                 '{0},128,156,1,128,156,0,148,236,0,148,236,1\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00002.jpg'))]
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
    def test_generate(self, m1, m2, m3):
        self.generator.generate()
        eq_(m3.call_args_list,
            [((os.path.join(self.output, 'test'), 'w'),),
             ((os.path.join(self.output, 'train'), 'w'),)])
        train = [(('{0},78,38,1,78,38,0,178,218,0,178,218,1\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0001.jpg')),),),
                 (('{0},128,156,0,128,156,1,148,236,1,148,236,0\n'.format(os.path.join(self.output, 'images', 'lsp_dataset', 'im0002.jpg')),),),
                 (('{0},78,38,0,78,38,1,178,218,1,178,218,0\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00001.jpg')),),),
                 (('{0},128,156,1,128,156,0,148,236,0,148,236,1\n'.format(os.path.join(self.output, 'images', 'lspet_dataset', 'im00002.jpg')),),)]
        eq_(m3.return_value.write.call_args_list, train)
