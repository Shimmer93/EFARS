#!/bin/bash

python train_estimator.py --model unipose --output_path UniPose-20-1e-3-plateau-sigma7 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 7
python train_estimator.py --model unipose --output_path UniPose-20-1e-3-plateau-sigma7 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 7
python train_estimator.py --model openpose --output_path OpenPose-20-1e-3-plateau-sigma3 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 3
python train_estimator.py --model openpose --output_path OpenPose-20-1e-3-plateau-sigma3 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 3