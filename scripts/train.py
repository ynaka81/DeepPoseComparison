# -*- coding: utf-8 -*-
""" Script for trianing pose net. """

import argparse
import sys

sys.path.append("./")
from modules.train.chainer import TrainPoseNet


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description='Training pose net for comparison \
        between chainer and pytorch about implementing DeepPose.')
    parser.add_argument(
        '--Nj', '-j', type=int, default=14, help='Number of joints.')
    parser.add_argument(
        '--epoch', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument(
        '--opt', '-o', type=str, default='MomentumSGD',
        choices=['MomentumSGD', 'Adam'], help='Optimization method.')
    parser.add_argument(
        '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU).')
    parser.add_argument(
        '--train', type=str, default='data/train', help='Path to training image-pose list file.')
    parser.add_argument(
        '--val', type=str, default='data/test', help='Path to validation image-pose list file.')
    parser.add_argument(
        '--batchsize', type=int, default=32, help='Learning minibatch size.')
    parser.add_argument(
        '--out', default='result', help='Output directory')
    parser.add_argument(
        '--resume', default=None,
        help='Initialize the trainer from given file. \
        The file name is "epoch-{.updater.epoch}.iter".')
    parser.add_argument(
        '--resume_model', type=str, default=None,
        help='Load model definition file to use for resuming training \
        (it\'s necessary when you resume a training). \
        The file name is "epoch-{.updater.epoch}.mode"')
    parser.add_argument(
        '--resume_opt', type=str, default=None,
        help='Load optimization states from this file \
        (it\'s necessary when you resume a training). \
        The file name is "epoch-{.updater.epoch}.state"')
    args = parser.parse_args()
    args_dict = vars(args)
    train = TrainPoseNet(**args_dict)
    train.start()

if __name__ == '__main__':
    main()
