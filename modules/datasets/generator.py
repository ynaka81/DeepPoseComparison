# -*- coding: utf-8 -*-
""" Generate dataset. """

import os
import cv2
import numpy as np
import tqdm

from modules.errors import SaveImageFailed
from modules.datasets.lsp.dataset import LSPDataset
from modules.datasets.lspet.dataset import LSPETDataset


class DatasetGenerator(object):
    """ Generate dataset from LSP and LSPET datasets.

    Args:
        image_size (int): Size of output image.
        crop_size (int): Size of cropping for DNN training.
        path (str): A path to download datasets.
        output (str): An output path for generated datasets.
    """

    def __init__(self, image_size=256, crop_size=227, path='orig_data', output='data'):
        try:
            os.makedirs(os.path.join(output, 'images'))
        except OSError:
            pass
        self.image_size = image_size
        self.crop_size = crop_size
        self.path = path
        self.output = output
        self.datasets = (LSPDataset(path), LSPETDataset(path))

    def _pad_image(self, image, joint):
        height, width, _ = image.shape
        shape = np.array((width, height))
        residual = (self.image_size - shape).clip(0, self.image_size)
        left, top = residual/2
        right, bottom = residual - residual/2
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        moved_joint = joint + (left, top, 0)
        return padded_image, moved_joint

    def _crop_image(self, image, joint):
        height, width, _ = image.shape
        shape = (width, height)
        visibility = joint[:, 2].astype(bool)
        visible_joint = joint[visibility]
        j_min = np.min(visible_joint, 0)
        j_max = np.max(visible_joint, 0)
        j_c = (j_min + j_max)/2
        crop_shape = [0, 0, 0, 0]
        # crop on a joint center
        for i in range(2):
            crop_shape[2*i] = max(0, int(j_c[i] - float(self.image_size)/2))
            crop_shape[2*i + 1] = min(shape[i], crop_shape[2*i] + self.image_size)
            crop_shape[2*i] -= self.image_size - (crop_shape[2*i + 1] - crop_shape[2*i])
        cropped_image = image[crop_shape[2]:crop_shape[3], crop_shape[0]:crop_shape[1]]
        moved_joint = joint - (crop_shape[0], crop_shape[2], 0)
        return cropped_image, moved_joint

    def _validate(self, joint):
        visibility = joint[:, 2].astype(bool)
        joint_xy = joint[visibility, :2]
        small_side_condition = (joint_xy > 0).all()
        big_side_condition = (joint_xy < self.image_size).all()
        if joint_xy.size > 0:
            j_min = np.min(joint_xy, 0)
            j_max = np.max(joint_xy, 0)
            j_diff = j_max - j_min
            crop_size_condition = (j_diff < self.crop_size).all()
        else:
            crop_size_condition = True
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

    def _make_dataset_line(self, image_path, joint):
        joint_list = ','.join(map(str, joint.flatten()))
        # format: image_filename, x_0, y_0, v_0, x_1, ...
        return image_path + ',' + joint_list + os.linesep

    def _generate_datasets(self):
        datasets = {'train': [], 'test': []}
        for dataset in self.datasets:
            print 'Generate dataset from {0}.'.format(dataset.name)
            # load dataset.
            dataset.load()
            # generate dataset
            for index in tqdm.trange(len(dataset), ascii=True):
                # get i-th data in the dataset.
                label, joint, image_file, image = dataset.get_data(index)
                # pad and crop image
                image, joint = self._pad_image(image, joint)
                image, joint = self._crop_image(image, joint)
                if not self._validate(joint):
                    continue
                # save the image
                image_path = self._save_image(dataset.name, image_file, image)
                # write database
                line = self._make_dataset_line(image_path, joint)
                datasets[label].append(line)
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
