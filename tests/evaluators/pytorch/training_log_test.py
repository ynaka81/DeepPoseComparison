# -*- coding: utf-8 -*-
# pylint: skip-file

import unittest
from nose.tools import eq_
from mock import patch

from modules.evaluators.pytorch import TrainingLog


class TestTrainingLog(unittest.TestCase):

    @patch('modules.evaluators.pytorch.training_log.open')
    def test_init(self, mock):
        # prepare mock.
        mock.return_value = [
            'elapsed_time: 4.2, loss: 0.23\n',
            'elapsed_time: 82.5, validation/loss: 0.1\n',
            'elapsed_time: 41.8, loss: 0.25'
        ]
        evaluator = TrainingLog('test_log')
        mock.assert_called_once_with('test_log')
        eq_(evaluator.t, [4.2, 41.8])
        eq_(evaluator.v, [0.23, 0.25])
