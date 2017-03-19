# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_
from mock import patch

from modules.evaluators.chainer import TrainingLog


class TestTrainingLog(unittest.TestCase):

    @patch('json.load')
    @patch('modules.evaluators.chainer.training_log.open', return_value='dummy_file')
    def test_init(self, m1, m2):
        # prepare mock.
        m2.return_value = [
            {
                "elapsed_time": 91.0,
                "iteration": 10,
                "main/loss": 0.36,
                "lr": 0.01,
                "epoch": 0
            },
            {
                "elapsed_time": 175.6,
                "iteration": 20,
                "main/loss": 0.22,
                "lr": 0.01,
                "epoch": 0
            }
        ]
        # test.
        evaluator = TrainingLog('test_log')
        m1.assert_called_once_with('test_log')
        m2.assert_called_once_with('dummy_file')
        eq_(evaluator.t, [91.0, 175.6])
        eq_(evaluator.v, [0.36, 0.22])
