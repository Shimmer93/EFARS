#!/bin/bash

#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-2-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-2
#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-3-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-3
#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-4-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-4
#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-3-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-3
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-2-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-2
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-3-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-3
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-4-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-4
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-5-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-5
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-2 --batch_size 8 --n_epochs 120 --max_lr 1e-2
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-3 --batch_size 8 --n_epochs 120 --max_lr 1e-3
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-4 --batch_size 8 --n_epochs 120 --max_lr 1e-4
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-5 --batch_size 8 --n_epochs 120 --max_lr 1e-5
#python train_2d_to_3d_temporal.py --model sem_gcn --output_path GCNTransformer-60-1e-2 --batch_size 8 --n_epochs 60 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model gcn_trans --output_path GCNTransformer-80-1e-2-hope --batch_size 8 --n_epochs 80 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model gcn_trans --output_path GCNTransformer-80-1e-3-hope --batch_size 8 --n_epochs 80 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model gcn_trans --output_path GCNTransformer-80-1e-4-hope --batch_size 8 --n_epochs 80 --max_lr 1e-4
#python train_2d_to_3d_temporal.py --model trans --output_path PureTransformer-80-1e-2 --batch_size 8 --n_epochs 80 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model trans --output_path PureTransformer-80-1e-3 --batch_size 8 --n_epochs 80 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model trans --output_path PureTransformer-80-1e-4 --batch_size 8 --n_epochs 80 --max_lr 1e-4
python train_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-2 --batch_size 8 --n_epochs 80 --max_lr 1e-2
python train_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-3 --batch_size 8 --n_epochs 80 --max_lr 1e-3
python train_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-4 --batch_size 8 --n_epochs 80 --max_lr 1e-4