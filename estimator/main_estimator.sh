#!/bin/bash

#python main_2d_estimator.py --model unipose --output_path UniPose-20-1e-2-plateau-sigma3-newgt --batch_size 2 --n_epochs 20 --lr 1e-2 --sigma 3
#python main_2d_estimator.py --model unipose --output_path UniPose-20-1e-3-plateau-sigma3-newgt --batch_size 2 --n_epochs 20 --lr 1e-3 --sigma 3
#python main_2d_estimator.py --model unipose --test --checkpoint /home/samuel/EFARS/estimator/checkpoints/UniPose-20-1e-2-plateau-sigma3-newgt/best-checkpoint-007epoch.bin --batch_size 2 --n_epochs 1 --lr 1e-4 --sigma 3
#python main_2d_estimator.py --model unipose --output_path UniPose-20-1e-3-plateau-sigma7 --batch_size 2 --n_epochs 20 --lr 1e-3 --sigma 7
#python main_2d_estimator.py --model openpose --output_path OpenPose-20-1e-3-plateau-sigma3 --batch_size 2 --n_epochs 20 --lr 1e-3 --sigma 3
#python main_2d_estimator.py --model openpose --output_path OpenPose-20-1e-3-plateau-sigma3 --batch_size 2 --n_epochs 20 --lr 1e-3 --sigma 3

#python main_2d_to_3d.py --model mlp --output_path Pose2Mesh-60-1e-2-aaaaa --batch_size 8 --n_epochs 60 --lr 1e-2
#python main_2d_to_3d.py --model mlp --output_path Pose2Mesh-60-1e-3-aaaaa --batch_size 8 --n_epochs 60 --lr 1e-3
#python main_2d_to_3d.py --model mlp --output_path Pose2Mesh-60-1e-4-aaaaa --batch_size 8 --n_epochs 60 --lr 1e-4
#python main_2d_to_3d.py --model mlp --output_path Pose2Mesh-60-1e-3-aaaaa --batch_size 8 --n_epochs 60 --lr 1e-3
#python main_2d_to_3d.py --model sem_gcn --output_path try --batch_size 32 --n_epochs 2 --lr 1e-2
#python main_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-3-aaa --batch_size 4 --n_epochs 120 --lr 1e-3
#python main_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-4-aaa --batch_size 4 --n_epochs 120 --lr 1e-4
#python main_2d_to_3d.py --model sem_gcn --output_path SemGCN-20-1e-3 --batch_size 32 --n_epochs 20 --lr 1e-3
python main_2d_to_3d.py --model videopose3d --output_path trytrytry --batch_size 32 --n_epochs 2 --lr 1e-2 --gpus 0 --test_after_train

#python main_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-3 --batch_size 16 --n_epochs 80 --lr 1e-3
#python main_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-4 --batch_size 16 --n_epochs 80 --lr 1e-4
#python main_2d_to_3d_temporal.py --model trans_enc --output_path PureTransformerModel-80-1e-3 --batch_size 16 --n_epochs 80 --lr 1e-3
#python main_2d_to_3d_temporal.py --model trans_enc --output_path PureTransformerModel-80-1e-4 --batch_size 16 --n_epochs 80 --lr 1e-4
#python main_2d_to_3d_temporal.py --model videopose3d --output_path tryatry --batch_size 32 --n_epochs 2 --lr 1e-2 --seq_len 32 --gpus 0 --test_after_train