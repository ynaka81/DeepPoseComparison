# Comparison between chainer and pytorch about implementing DeepPose

[![CircleCI](https://circleci.com/gh/ynaka81/DeepPoseComparison.svg?style=svg)](https://circleci.com/gh/ynaka81/DeepPoseComparison)

This repository contains codes for Qiita blog [post](http://qiita.com/ynaka81/items/85659dff4d1c2c593f21).

## Usage

### Dataset preparation
First download and prepare datasets for training by running the following script.
```
python scripts/dataset.py
```

### Start training
If you want to train model with Chainer, just run:
```
python scripts/train.py chainer
```
In the case of PyTorch:
```
python scripts/train.py pytorch
```
If you want to run `train.py` with your own settings, please check the options by `python scripts/train.py --help` and give customized training settings.

### Visualize
Visualizing training time, just run:
```
python scripts/plot_training_time.py "title of graph"
```

Visualizing inference time, just run:
```
python scripts/plot_estimating_time.py 10000 "title of graph"
```

## Install
```
pip install pyflakes pylint
pip install sphinx
pip install -r requirements.txt
```
