#!/bin/bash

#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-2-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-2
#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-3-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-3
#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-4-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-4
#python train_2d_to_3d.py --model pose2mesh --output_path Pose2Mesh-60-1e-3-aaaaa --batch_size 8 --n_epochs 60 --max_lr 1e-3
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-2-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-2
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-3-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-3
#python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-120-1e-4-aaa --batch_size 4 --n_epochs 120 --max_lr 1e-4
python train_2d_to_3d.py --model sem_gcn --output_path SemGCN-60-1e-3-maybe --batch_size 16 --n_epochs 60 --max_lr 1e-3
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-2 --batch_size 8 --n_epochs 120 --max_lr 1e-2
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-3 --batch_size 8 --n_epochs 120 --max_lr 1e-3
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-4 --batch_size 8 --n_epochs 120 --max_lr 1e-4
#python train_2d_to_3d.py --model gcn --output_path GCN-120-1e-5 --batch_size 8 --n_epochs 120 --max_lr 1e-5
#python train_2d_to_3d_temporal.py --model sem_gcn --output_path GCNTransformer-60-1e-2 --batch_size 8 --n_epochs 60 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model gcn_trans --output_path GCNTransformer-20-1e-2 --batch_size 32 --n_epochs 20 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model gcn_trans --output_path GCNTransformer-20-1e-3 --batch_size 32 --n_epochs 20 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model gcn_trans --output_path GCNTransformer-20-1e-4 --batch_size 32 --n_epochs 20 --max_lr 1e-4
#python train_2d_to_3d_temporal.py --model trans --output_path PureTransformer-20-1e-2 --batch_size 32 --n_epochs 20 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model trans --output_path PureTransformer-20-1e-3 --batch_size 32 --n_epochs 20 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model trans --output_path PureTransformer-20-1e-4 --batch_size 32 --n_epochs 20 --max_lr 1e-4
#python train_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-3 --batch_size 16 --n_epochs 80 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model gcn_trans_enc --output_path GCNTransformerModel-80-1e-4 --batch_size 16 --n_epochs 80 --max_lr 1e-4
#python train_2d_to_3d_temporal.py --model trans_enc --output_path PureTransformerModel-80-1e-3 --batch_size 16 --n_epochs 80 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model trans_enc --output_path PureTransformerModel-80-1e-4 --batch_size 16 --n_epochs 80 --max_lr 1e-4
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-20-1e-2 --batch_size 32 --n_epochs 20 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-20-1e-3 --batch_size 32 --n_epochs 20 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-20-1e-4-continue2 --batch_size 32 --n_epochs 60 --max_lr 1e-5
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-20-1e-2 --batch_size 32 --n_epochs 20 --max_lr 1e-2
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-80-1e-3 --batch_size 32 --n_epochs 80 --max_lr 1e-3
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-80-1e-4 --batch_size 32 --n_epochs 80 --max_lr 1e-4
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-20-1e-3-dim512 --batch_size 32 --n_epochs 20 --max_lr 1e-3 --hid_dim 512
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-20-1e-4-dim512 --batch_size 32 --n_epochs 20 --max_lr 1e-4 --hid_dim 512
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-80-1e-3-dim512 --batch_size 32 --n_epochs 80 --max_lr 1e-3 --hid_dim 512
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-80-1e-4-dim512 --batch_size 32 --n_epochs 80 --max_lr 1e-4 --hid_dim 512
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-20-1e-4-dim512-reg3 --batch_size 32 --n_epochs 20 --max_lr 1e-4 --hid_dim 512
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-20-1e-4-dim2048 --batch_size 32 --n_epochs 20 --max_lr 1e-4 --hid_dim 2048
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-80-1e-3-dim2048 --batch_size 32 --n_epochs 80 --max_lr 1e-3 --hid_dim 2048
#python train_2d_to_3d_temporal.py --model mlp_lstm --output_path MLPLSTM-80-1e-4-dim2048 --batch_size 32 --n_epochs 80 --max_lr 1e-4 --hid_dim 2048
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-20-1e-3-dim512 --batch_size 32 --n_epochs 20 --max_lr 1e-3 --hid_dim 512
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-20-1e-4-dim512 --batch_size 32 --n_epochs 20 --max_lr 1e-4 --hid_dim 512
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-80-1e-3-dim512 --batch_size 32 --n_epochs 80 --max_lr 1e-3 --hid_dim 512
#python train_2d_to_3d_temporal.py --model gcn_lstm --output_path GCNLSTM-80-1e-4-dim512 --batch_size 32 --n_epochs 80 --max_lr 1e-4 --hid_dim 512