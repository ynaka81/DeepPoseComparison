# -*- coding: utf-8 -*-
""" Generate LSP dataset. """

import os
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from modules.errors import FileNotFoundError, SaveImageFailed, CropFailed


class LSPDatasetGenerator(object):
    """ Generate LSP dataset from
    'Leeds Sports Pose Dataset' and 'Leeds Sports Pose Extended Training Dataset'

    Args:
        ksize (int): Size of filter.
        stride (int): Stride of filter applications.
        path (str): A path to download datasets.
        output (str): An output path for generated datasets.
    """

    def __init__(self, ksize=11, stride=4, path='orig_data', output='data'):
        """ Constructor of generator. """
        try:
            os.makedirs(os.path.join(output, 'images'))
        except OSError:
            pass
        self.ksize = ksize
        self.stride = stride
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

    def _validate_crop_feasibility(self, shape, j_min, j_max):
        crop = (shape - self.ksize)%self.stride
        # raise expection when joints in image are too tight to crop
        if j_min == 0 or j_max >= shape - 1:
            raise CropFailed('Joints in the image are too tight to crop.')
        residual = shape - (np.ceil(j_max - j_min) + 3)
        if residual < crop:
            raise CropFailed('Joints in the image are too tight to crop.')
        return crop

    def _crop_image(self, image, joint):
        j_min = np.min(joint, 0)
        j_max = np.max(joint, 0)
        height, width, _ = image.shape
        shape = (width, height)
        crop_shape = [0]*4
        # calculate crop value
        for i in range(2):
            # validate feasibility of cropping
            crop = self._validate_crop_feasibility(shape[i], j_min[i], j_max[i])
            # crop image in the side which has more margin
            if shape[i] < int(np.ceil(j_min[i])) + int(np.floor(j_max[i])) + 1:
                crop_shape[2*i] = min(crop, int(np.floor(j_min[i])) - 1)
                crop_shape[2*i + 1] = shape[i] - crop + crop_shape[2*i]
            else:
                crop_shape[2*i + 1] = max(shape[i] - crop, int(np.ceil(j_max[i])) + 2)
                crop_shape[2*i] = crop_shape[2*i + 1] - (shape[i] - crop)
        return image[crop_shape[2]:crop_shape[3], crop_shape[0]:crop_shape[1]]

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
                # save cropped image
                try:
                    image = self._crop_image(image, joint)
                except CropFailed:
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
