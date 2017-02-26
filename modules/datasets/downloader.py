# -*- coding: utf-8 -*-
""" Download LSP dataset. """

import os
import zipfile
import wget


class LSPDatasetDownloader(object):
    """ Download LSP dataset:
    'Leeds Sports Pose Dataset' and 'Leeds Sports Pose Extended Training Dataset'

    Args:
        path (str): A path to download datasets.
    """

    def __init__(self, path='orig_data'):
        """ Constructor of downloader. """
        try:
            os.makedirs(path)
        except OSError:
            pass
        self.path = path
        self.dataset = [
            ('lsp_dataset', 'http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip'),
            ('lspet_dataset', 'http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip')]

    def _need_download(self, dataset_name):
        # download when file doesn't exist.
        path = os.path.join(self.path, dataset_name)
        return not os.path.isdir(path)

    def _download_dataset(self, url):
        path = wget.download(url, self.path)
        return path

    def _extract_dataset(self, dataset_name, path):
        extract_path = self.path
        if dataset_name == 'lspet_dataset':
            extract_path = os.path.join(extract_path, dataset_name)
        with zipfile.ZipFile(path, "r") as zip_file:
            zip_file.extractall(extract_path)

    # pylint: disable=no-self-use
    def _cleanup(self, path):
        os.remove(path)

    def download(self):
        """ Download LSP dataset. """
        for dataset_name, url in self.dataset:
            if not self._need_download(dataset_name):
                continue
            path = self._download_dataset(url)
            self._extract_dataset(dataset_name, path)
            self._cleanup(path)
