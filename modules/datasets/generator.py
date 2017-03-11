# -*- coding: utf-8 -*-
""" Generate LSP dataset. """

import os
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from modules.errors import FileNotFoundError, SaveImageFailed


class LSPDatasetGenerator(object):
    """ Generate LSP dataset from
    'Leeds Sports Pose Dataset' and 'Leeds Sports Pose Extended Training Dataset'

    Args:
        image_size (int): Size of output image.
        crop_size (int): Size of cropping for DNN training.
        path (str): A path to download datasets.
        output (str): An output path for generated datasets.
    """

    def __init__(self, image_size=256, crop_size=227, path='orig_data', output='data'):
        """ Constructor of generator. """
        try:
            os.makedirs(os.path.join(output, 'images'))
        except OSError:
            pass
        self.image_size = image_size
        self.crop_size = crop_size
        self.path = path
        self.output = output
        self.dataset = ['lsp_dataset', 'lspet_dataset']

    def _load_joints(self, dataset_name):
        joints = loadmat(os.path.join(self.path, dataset_name, 'joints.mat'))['joints']
        if dataset_name == 'lsp_dataset':
            joints = joints.transpose(2, 1, 0)
            joints[:, :, 2] = np.logical_not(joints[:, :, 2]).astype(int)
        else:
            joints = joints.transpose(2, 0, 1)
        return joints

    def _load_image(self, dataset_name, i):
        if dataset_name == 'lsp_dataset':
            image_file = 'im{0:04d}.jpg'.format(i)
        else:
            image_file = 'im{0:05d}.jpg'.format(i)
        path = os.path.join(self.path, dataset_name, 'images', image_file)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError('{0} is not found.'.format(path))
        return image_file, image

    def _scale_image(self, image, joint):
        height, width, _ = image.shape
        # scale image so that smaller side is equal to desired image size
        small_side = min(width, height)
        ratio = float(self.image_size)/small_side
        return cv2.resize(image, None, fx=ratio, fy=ratio), joint*(ratio, ratio, 1)

    def _crop_image(self, image, joint):
        # crop bigger side
        index = 1 - np.argmax(image.shape)
        # crop on a joint center
        j_c = np.mean(joint[:, index])
        left_top = max(0, int(j_c - float(self.image_size)/2))
        right_bottom = min(image.shape[1 - index], left_top + self.image_size)
        left_top -= self.image_size - (right_bottom - left_top)
        if index == 0:
            return image[:, left_top:right_bottom, :], joint - (left_top, 0, 0)
        else:
            return image[left_top:right_bottom, :, :], joint - (0, left_top, 0)

    def _validate(self, joint):
        joint_xy = joint[:, :2]
        small_side_condition = (joint_xy > 0).all()
        big_side_condition = (joint_xy < self.image_size).all()
        j_min = np.min(joint_xy, 0)
        j_max = np.max(joint_xy, 0)
        j_diff = j_max - j_min
        crop_size_condition = (j_diff < self.crop_size).all()
        return small_side_condition and big_side_condition and crop_size_condition

    def _save_image(self, dataset_name, image_file, image):
        path = os.path.join(self.output, 'images', dataset_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        image_path = os.path.join(path, image_file)
        ret = cv2.imwrite(image_path, image)
        if not ret:
            raise SaveImageFailed('Failed to save {0}.'.format(image_path))
        return image_path

    # pylint: disable=no-self-use
    def _get_data_label(self, dataset_name, i):
        if dataset_name == 'lsp_dataset' and i > 1000:
            return 'test'
        return 'train'

    # pylint: disable=no-self-use
    def _make_dataset_line(self, image_path, joint):
        joint_list = ','.join(map(str, joint.flatten()))
        # format: image_filename, x_0, y_0, v_0, x_1, ...
        return image_path + ',' + joint_list + os.linesep

    def _generate_datasets(self):
        datasets = {'train': [], 'test': []}
        for dataset_name in self.dataset:
            print 'Generate dataset from {0}.'.format(dataset_name)
            # load dataset
            joints = self._load_joints(dataset_name)
            # generate dataset
            for i, joint in enumerate(tqdm(joints, ascii=True), 1):
                # load image
                image_file, image = self._load_image(dataset_name, i)
                # save scaled & cropped image
                image, joint = self._scale_image(image, joint)
                image, joint = self._crop_image(image, joint)
                if not self._validate(joint):
                    continue
                image_path = self._save_image(dataset_name, image_file, image)
                # write datase
                line = self._make_dataset_line(image_path, joint)
                datasets[self._get_data_label(dataset_name, i)].append(line)
        return datasets

    def _write_datasets(self, datasets):
        for name, lines in datasets.items():
            output = open(os.path.join(self.output, name), 'w')
            for line in lines:
                output.write(line)

    def generate(self):
        """ Generate LSP dataset. """
        datasets = self._generate_datasets()
        self._write_datasets(datasets)
