#!/bin/bash

#python train_skeleton_classifier.py --model st_gcn --output_path ST-GCN-20-1e-2-nontemporal --batch_size 32 --n_epochs 20 --max_lr 1e-2
#python train_skeleton_classifier.py --model st_gcn --output_path ST-GCN-20-1e-3-nontemporal --batch_size 32 --n_epochs 20 --max_lr 1e-3
python train_skeleton_classifier.py --model st_gcn --output_path ST-GCN-20-1e-2-drop0.1 --batch_size 32 --n_epochs 20 --max_lr 1e-2
#python train_skeleton_classifier.py --model ctr_gcn --output_path CTR_GCN-20-1e-2 --batch_size 16 --n_epochs 20 --max_lr 1e-2
#python train_skeleton_classifier.py --model ctr_gcn --output_path CTR_GCN-20-1e-3 --batch_size 16 --n_epochs 20 --max_lr 1e-3
#python train_skeleton_classifier.py --model ctr_gcn --output_path CTR_GCN-20-1e-4 --batch_size 16 --n_epochs 20 --max_lr 1e-4
#python train_skeleton_classifier.py --model mlp --output_path MLP-20-1e-3 --batch_size 32 --n_epochs 20 --max_lr 1e-3
#python train_skeleton_classifier.py --model mlp_trans_enc --output_path MLPTransformerModel-20-1e-4 --batch_size 32 --n_epochs 20 --max_lr 1e-4