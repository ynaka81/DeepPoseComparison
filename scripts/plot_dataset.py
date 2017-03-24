# -*- coding: utf-8 -*-
""" Script for visualizing dataset. """

import sys
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('./')
from modules.dataset_indexing.chainer import PoseDataset


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description='Visualize image and its pose in dataset.')
    parser.add_argument(
        'path', type=str, help='Path to dataset (image-pose list file).')
    parser.add_argument(
        '--out', default='result/dataset', help='Output directory.')
    parser.add_argument(
        '--use-visibility', '-v', action='store_true', help='Use visibility to plot pose.')
    args = parser.parse_args()
    output_dir = os.path.join(args.out, os.path.basename(args.path))
    # create directory.
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    # get dataset.
    dataset = PoseDataset(args.path, data_augmentation=False)
    for index, (image, pose, visibility) in enumerate(tqdm(dataset, ascii=True)):
        # get data.
        _, size, _ = image.shape
        if args.use_visibility:
            pose = pose[visibility.ravel().astype(bool)]
        pose *= size
        pose_x, pose_y = zip(*pose)
        # plot image and pose.
        fig = plt.figure()
        plt.imshow(image.transpose(1, 2, 0), vmin=0., vmax=1.)
        plt.scatter(pose_x, pose_y, color="r", s=5)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, '{}.png'.format(index)))
        plt.close(fig)

if __name__ == '__main__':
    main()
