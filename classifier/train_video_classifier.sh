#!/bin/bash

#python train_video_classifier.py --model mobilenetv2 --output_path MobileNetv2-60-1e-2 --batch_size 16 --n_epochs 60 --max_lr 1e-2
#python train_video_classifier.py --model mobilenetv2 --output_path MobileNetv2-60-1e-4 --batch_size 4 --n_epochs 60 --max_lr 1e-4
#python train_video_classifier.py --model cnnlstm --output_path CNNLSTM-60-1e-2 --batch_size 16 --n_epochs 60 --max_lr 1e-2
#python train_video_classifier.py --model cnnlstm --output_path CNNLSTM-60-1e-3 --batch_size 16 --n_epochs 60 --max_lr 1e-3
#python train_video_classifier.py --model cnnlstm --output_path CNNLSTM-30-1e-4 --batch_size 16 --n_epochs 30 --max_lr 1e-4
#python train_video_classifier.py --model timesformer --output_path TimeSformer-60-1e-2 --batch_size 16 --n_epochs 60 --max_lr 1e-2
#python train_video_classifier.py --model timesformer --output_path TimeSformer-30-1e-3 --batch_size 16 --n_epochs 30 --max_lr 1e-3
python train_video_classifier.py --model timesformer --output_path TimeSformer-30-1e-4-moredata --batch_size 16 --n_epochs 30 --max_lr 1e-4
#python train_video_classifier.py --model mobilenetv2 --output_path MobileNetv2-30-1e-4 --batch_size 16 --n_epochs 30 --max_lr 1e-4
#python train_video_classifier.py --model shufflenetv2 --output_path ShuffleNetv2-30-1e-4 --batch_size 16 --n_epochs 30 --max_lr 1e-4
