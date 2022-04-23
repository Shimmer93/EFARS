#!/bin/bash

#python train_estimator.py --model unipose --output_path UniPose-20-1e-2-plateau-sigma3-newgt --batch_size 2 --n_epochs 20 --max_lr 1e-2 --sigma 3
#python train_estimator.py --model unipose --output_path UniPose-20-1e-3-plateau-sigma3-newgt --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 3
python train_estimator.py --model unipose --output_path try --batch_size 2 --n_epochs 1 --lr 1e-4 --sigma 3
#python train_estimator.py --model unipose --output_path UniPose-20-1e-3-plateau-sigma7 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 7
#python train_estimator.py --model openpose --output_path OpenPose-20-1e-3-plateau-sigma3 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 3
#python train_estimator.py --model openpose --output_path OpenPose-20-1e-3-plateau-sigma3 --batch_size 2 --n_epochs 20 --max_lr 1e-3 --sigma 3