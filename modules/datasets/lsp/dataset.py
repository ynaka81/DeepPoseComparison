# -*- coding: utf-8 -*-
""" LSP Dataset. """

import os
import cv2
import numpy as np
from scipy.io import loadmat

from modules.datasets.common.dataset import Dataset


class LSPDataset(Dataset):
    """ LSP dataset: 'Leeds Sports Pose Dataset'.

    Args:
        path (str): A path to download datasets.
    """

    def __init__(self, path='orig_data'):
        super(LSPDataset, self).__init__(
            'lsp_dataset',
            'http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip', path)

    def _get_extract_path(self):
        return self.path

    def _load_joints(self):
        path = os.path.join(self.path, self.name, 'joints.mat')
        raw_joints = loadmat(path)['joints']
        joints = raw_joints.transpose(2, 1, 0)
        joints[:, :, 2] = np.logical_not(joints[:, :, 2]).astype(int)
        return joints

    def _get_image(self, i):
        image_file = 'im{0:04d}.jpg'.format(i + 1)
        path = os.path.join(self.path, self.name, 'images', image_file)
        image = cv2.imread(path)
        return image_file, image

    def _get_data_label(self, i):
        if i > 1000:
            return 'test'
        return 'train'
