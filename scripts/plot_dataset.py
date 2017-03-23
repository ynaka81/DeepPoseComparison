# -*- coding: utf-8 -*-
""" Script for visualizing dataset. """

import sys
import matplotlib.pyplot as plt

sys.path.append('./')
from modules.dataset_indexing.chainer import PoseDataset

def main():
    """ Main function. """
    dataset = PoseDataset('data/test', data_augmentation=False)
    for index, (image, pose, _) in enumerate(dataset):
        # get data.
        _, size, _ = image.shape
        pose *= size
        pose_x, pose_y = zip(*pose)
        # plot image and pose.
        plt.figure()
        plt.imshow(image.transpose(1, 2, 0), vmin=0., vmax=1.)
        plt.scatter(pose_x, pose_y, color="r", s=5)
        plt.axis("off")
        plt.savefig('result/dataset/{}.png'.format(index))

if __name__ == '__main__':
    main()
