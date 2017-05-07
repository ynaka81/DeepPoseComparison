# -*- coding: utf-8 -*-
""" Script for plotting core process (forward + backward) time of pose net. """

import argparse
import sys

sys.path.append("./")
from modules.evaluators import CoreProcessTimeEvaluator


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description='Core process (forward + backward) time comparison of pose net between chainer and pytorch.')
    parser.add_argument(
        'samples', type=int, help='Samples of comparison.')
    parser.add_argument(
        'title', type=str, help='Title of comparison graph.')
    parser.add_argument(
        '--max-batch-index', '-m', type=int, default=5, help='Index of max batch size: 2^m.')
    parser.add_argument(
        '--only-inference', action='store_true', help='If true, core process includes only inference process.')
    parser.add_argument(
        '--Nj', '-j', type=int, default=14, help='Number of joints.')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU).')
    parser.add_argument(
        '--chainer-model-file', '-c', type=str,
        default='result/chainer/epoch-100.model', help='Chainer model parameter file.')
    parser.add_argument(
        '--pytorch-model-file', '-p', type=str,
        default='result/pytorch/epoch-100.model', help='Pytorch model parameter file.')
    parser.add_argument(
        '--filename', '-f', type=str, default='data/train', help='Image-pose list file.')
    parser.add_argument(
        '--output', '-o', type=str, default='result', help='Output directory.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    args = parser.parse_args()
    args_dict = vars(args)
    evaluator = CoreProcessTimeEvaluator(**args_dict)
    evaluator.plot(args.samples, args.title)


if __name__ == '__main__':
    main()
