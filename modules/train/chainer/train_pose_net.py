# -*- coding: utf-8 -*-
""" Train pose net. """

import os
import chainer
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import serializers

from modules.errors import FileNotFoundError, UnknownOptimizationMethodError
from modules.models.chainer import AlexNet
from modules.dataset_indexing.chainer import PoseDataset


class TestModeEvaluator(extensions.Evaluator):
    """ The supplement class for validating the pose net.
    """

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

# pylint: disable=too-many-instance-attributes
class TrainPoseNet(object):
    """ Train pose net of estimating 2D pose from image.

    Args:
        Nj (int): Number of joints.
        epoch (int): Number of epochs to train.
        opt (str): Optimization method.
        gpu (int): GPU ID (negative value indicates CPU).
        train (str): Path to training image-pose list file.
        val (str): Path to validation image-pose list file.
        batchsize (int): Learning minibatch size.
        out (str): Output directory.
        resume (str): Initialize the trainer from given file.
            The file name is 'epoch-{.updater.epoch}.iter'.
        resume_model (str): Load model definition file to use for resuming training
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{.updater.epoch}.model'.
        resume_opt (str): Load optimization states from this file
            (it\'s necessary when you resume a training).
            The file name is 'epoch-{.updater.epoch}.state'.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, Nj=14, epoch=100, gpu=-1, opt='Adam',
                 train='data/train', val='data/test', batchsize=32, out='result',
                 resume=None, resume_model=None, resume_opt=None):
        self.Nj = Nj
        self.epoch = epoch
        self.gpu = gpu
        self.opt = opt
        self.train = train
        self.val = val
        self.batchsize = batchsize
        self.out = out
        self.resume = resume
        self.resume_model = resume_model
        self.resume_opt = resume_opt
        # validate arguments.
        self._validate_arguments()

    def _validate_arguments(self):
        for path in (self.train, self.val):
            if not os.path.isfile(path):
                raise FileNotFoundError('{0} is not found.'.format(path))
        if self.opt not in ('MomentumSGD', 'Adam'):
            raise UnknownOptimizationMethodError(
                '{0} is unknown optimization method.'.format(self.opt))
        if self.resume is not None:
            for path in (self.resume, self.resume_model, self.resume_opt):
                if not os.path.isfile(path):
                    raise FileNotFoundError('{0} is not found.'.format(path))

    def _get_optimizer(self):
        if self.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD()
        elif self.opt == "Adam":
            optimizer = optimizers.Adam()
        return optimizer

    def start(self):
        """ Train pose net. """
        # initialize model to train.
        model = AlexNet(self.Nj)
        if self.resume_model:
            serializers.load_npz(self.resume_model, model)
        # prepare gpu.
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            model.to_gpu()
        # load the datasets.
        train = PoseDataset(self.train)
        val = PoseDataset(self.val, data_augmentation=False)
        # training/validation iterators.
        train_iter = chainer.iterators.MultiprocessIterator(
            train, self.batchsize)
        val_iter = chainer.iterators.MultiprocessIterator(
            val, self.batchsize, repeat=False, shuffle=False)
        # Set up an optimizer
        optimizer = self._get_optimizer()
        optimizer.setup(model)
        if self.resume_opt:
            chainer.serializers.load_npz(self.resume_opt, optimizer)
        # Set up a trainer
        updater = training.StandardUpdater(train_iter, optimizer, device=self.gpu)
        trainer = training.Trainer(
            updater, (self.epoch, 'epoch'), os.path.join(self.out, 'chainer'))
        # standard trainer settings
        trainer.extend(extensions.dump_graph('main/loss'))
        val_interval = (10, 'epoch')
        trainer.extend(TestModeEvaluator(val_iter, model, device=self.gpu), trigger=val_interval)
        # save parameters and optimization state per validation step
        resume_interval = (self.epoch/10, 'epoch')
        trainer.extend(extensions.snapshot_object(
            model, "epoch-{.updater.epoch}.model"), trigger=resume_interval)
        trainer.extend(extensions.snapshot_object(
            optimizer, "epoch-{.updater.epoch}.state"), trigger=resume_interval)
        trainer.extend(extensions.snapshot(
            filename="epoch-{.updater.epoch}.iter"), trigger=resume_interval)
        # show log
        log_interval = (10, "iteration")
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'lr']), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        # start training
        if self.resume:
            chainer.serializers.load_npz(self.resume, trainer)
        trainer.run()
