# -*- coding: utf-8 -*-
""" Script for downloading and generating dataset. """

import argparse
import sys

sys.path.append("./")
from modules.datasets import LSPDatasetDownloader, LSPDatasetGenerator


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description="Generating LSP dataset for comparison \
        between chainer and pytorch about implementing DeepPose.")
    parser.add_argument("--ksize", "-k", type=int, default=11, help="Size of filter.")
    parser.add_argument(
        "--stride", "-s", type=int, default=4, help="Stride of filter applications.")
    parser.add_argument(
        "--path", "-p", type=str, default="orig_data", help="A path to download datasets.")
    parser.add_argument(
        "--output", "-o", type=str, default="data", help="An output path for generated datasets.")
    # main process
    args = parser.parse_args()
    downloader = LSPDatasetDownloader(args.path)
    generator = LSPDatasetGenerator(args.ksize, args.stride, args.path, args.output)
    downloader.download()
    generator.generate()

if __name__ == '__main__':
    main()
