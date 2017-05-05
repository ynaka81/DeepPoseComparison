# -*- coding: utf-8 -*-
""" Script for downloading and generating dataset. """

import argparse
import sys

sys.path.append("./")
from modules.datasets import DatasetGenerator


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description="Generating LSP dataset for comparison \
        between chainer and pytorch about implementing DeepPose.")
    parser.add_argument(
        "--image_size", "-S", type=int, default=256, help="Size of output image.")
    parser.add_argument(
        "--crop_size", "-C", type=int, default=227, help="Size of cropping for DNN training.")
    parser.add_argument(
        "--path", "-p", type=str, default="orig_data", help="A path to download datasets.")
    parser.add_argument(
        "--output", "-o", type=str, default="data", help="An output path for generated datasets.")
    # main process
    args = parser.parse_args()
    generator = DatasetGenerator(args.image_size, args.crop_size, args.path, args.output)
    generator.generate()


if __name__ == '__main__':
    main()
