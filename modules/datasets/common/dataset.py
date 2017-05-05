# -*- coding: utf-8 -*-
""" Dataset. """

import os
import zipfile
import wget

from modules.errors import FileNotFoundError


class Dataset(object):
    """ Basic class of dataset.

    Args:
        name (str): Name of dataset.
        url (str): URL of dataset.
        path (str): A path to download datasets.
    """

    def __init__(self, name, url, path='orig_data'):
        try:
            os.makedirs(path)
        except OSError:
            pass
        self.name = name
        self.url = url
        self.path = path
        self.joints = None

    def load(self):
        """ Load a dataset.
        If a dataset has not been downloaded yet, this class downloads it.
        """
        # download a dataset if needed.
        self._download()
        # load joints.
        self.joints = self._load_joints()

    def get_data(self, i):
        """ Get i-th data (joint and image).

        Args:
            i (int): Index of data.

        Returns:
            A tuple of data. (label, joint, image)
        """
        label = self._get_data_label(i)
        joint = self.joints[i]
        image_file, image = self._get_image(i)
        if image is None:
            raise FileNotFoundError('{0} is not found.'.format(image_file))
        return label, joint, image_file, image

    def __len__(self):
        return len(self.joints)

    def _download(self):
        # download when file doesn't exist.
        path = os.path.join(self.path, self.name)
        if not os.path.isdir(path):
            # download a dataset.
            path = wget.download(self.url, self.path)
            # extract the dataset.
            with zipfile.ZipFile(path, 'r') as zip_file:
                zip_file.extractall(self._get_extract_path())
            # remove downloaded zip file.
            os.remove(path)

    def _get_extract_path(self):
        raise NotImplementedError

    def _load_joints(self):
        raise NotImplementedError

    def _get_image(self, i):
        raise NotImplementedError

    def _get_data_label(self, i):
        raise NotImplementedError
