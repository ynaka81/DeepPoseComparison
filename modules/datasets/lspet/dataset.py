# -*- coding: utf-8 -*-
""" LSPET Dataset. """

import os
import cv2
from scipy.io import loadmat

from modules.datasets.common.dataset import Dataset


class LSPETDataset(Dataset):
    """ LSPET dataset: 'Leeds Sports Pose Extended Training Dataset'.

    Args:
        path (str): A path to download datasets.
    """

    def __init__(self, path='orig_data'):
        super(LSPETDataset, self).__init__(
            'lspet_dataset',
            'http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip', path)

    def _get_extract_path(self):
        return os.path.join(self.path, self.name)

    def _load_joints(self):
        path = os.path.join(self.path, self.name, 'joints.mat')
        raw_joints = loadmat(path)['joints']
        joints = raw_joints.transpose(2, 0, 1)
        return joints

    def _get_image(self, i):
        image_file = 'im{0:05d}.jpg'.format(i + 1)
        path = os.path.join(self.path, self.name, 'images', image_file)
        image = cv2.imread(path)
        return image_file, image

    def _get_data_label(self, i):
        return 'train'
