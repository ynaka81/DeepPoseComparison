# -*- coding: utf-8 -*-
""" AlexNet implementation. """

import chainer
import chainer.functions as F
import chainer.links as L

from modules.functions.chainer import mean_squared_error


class AlexNet(chainer.Chain):
    """ The AlexNet :
    'A. Krizhevsky, I. Sutskever, and G. Hinton.
    Imagenet clas-sification with deep convolutional neural networks. InNIPS , 2012'

    Args:
        Nj (int): Size of joints.
        use_visibility (bool): When it is ``True``,
            the function uses visibility to compute mean squared error.
    """

    def __init__(self, Nj, use_visibility=False):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(None, 96, 11, stride=4),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 384, 3, pad=1),
            conv4=L.Convolution2D(None, 384, 3, pad=1),
            conv5=L.Convolution2D(None, 256, 3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, Nj*2),
        )
        self.Nj = Nj
        self.use_visibility = use_visibility
        self.train = True

    # pylint: disable=no-member
    def predict(self, x):
        """ Predict 2D pose from image. """
        # layer1
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        # layer2
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        # layer3-5
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        # layer6-8
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return F.reshape(h, (-1, self.Nj, 2))

    def __call__(self, image, x, v):
        y = self.predict(image)
        loss = mean_squared_error(y, x, v, use_visibility=self.use_visibility)
        chainer.report({'loss': loss}, self)
        return loss
