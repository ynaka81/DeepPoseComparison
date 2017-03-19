# -*- coding: utf-8 -*-
""" Script for plotting estimating time of pose net. """

import argparse
import sys

sys.path.append("./")
from modules.evaluators import EstimatingTimeEvaluator


def main():
    """ Main function. """
    # arg definition
    parser = argparse.ArgumentParser(
        description='Estimating time comparison of pose net between chainer and pytorch.')
    parser.add_argument(
        'title', type=str, help='Title of comparison graph.')
    parser.add_argument(
        '--batchsize', '-b', type=int, default=1, help='Estimating batch size.')
    parser.add_argument(
        '--output', '-o', type=str, default='result', help='Output directory.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    args = parser.parse_args()
    evaluator = EstimatingTimeEvaluator(args.batchsize, args.output)
    evaluator.plot(args.title, args.debug)

if __name__ == '__main__':
    main()
