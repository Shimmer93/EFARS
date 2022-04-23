#!/bin/bash

#python main_video_classifier.py --model mobilenetv2 --output_path MobileNetv2-60-1e-2 --batch_size 16 --n_epochs 60 --lr 1e-2
#python main_video_classifier.py --model mobilenetv2 --output_path MobileNetv2-60-1e-4 --batch_size 4 --n_epochs 60 --lr 1e-4
#python main_video_classifier.py --model cnnlstm --output_path CNNLSTM-60-1e-2 --batch_size 16 --n_epochs 60 --lr 1e-2
#python main_video_classifier.py --model cnnlstm --output_path CNNLSTM-60-1e-3 --batch_size 16 --n_epochs 60 --lr 1e-3
#python main_video_classifier.py --model cnnlstm --output_path CNNLSTM-30-1e-4 --batch_size 16 --n_epochs 30 --lr 1e-4
#python main_video_classifier.py --model timesformer --output_path TimeSformer-60-1e-2 --batch_size 16 --n_epochs 60 --lr 1e-2
#python main_video_classifier.py --model timesformer --output_path TimeSformer-30-1e-3 --batch_size 16 --n_epochs 30 --lr 1e-3
python main_video_classifier.py --model timesformer --output_path try --batch_size 16 --n_epochs 2 --lr 1e-4
#python main_video_classifier.py --model mobilenetv2 --output_path MobileNetv2-30-1e-4 --batch_size 16 --n_epochs 30 --lr 1e-4
#python main_video_classifier.py --model shufflenetv2 --output_path ShuffleNetv2-30-1e-4 --batch_size 16 --n_epochs 30 --lr 1e-4

#python main_skeleton_classifier.py --model st_gcn --output_path ST-GCN-20-1e-2-nontemporal --batch_size 32 --n_epochs 20 --lr 1e-2
#python main_skeleton_classifier.py --model st_gcn --output_path ST-GCN-20-1e-3-nontemporal --batch_size 32 --n_epochs 20 --lr 1e-3
python main_skeleton_classifier.py --model st_gcn --output_path trytry --batch_size 32 --n_epochs 2 --lr 1e-2
#python main_skeleton_classifier.py --model ctr_gcn --output_path CTR_GCN-20-1e-2 --batch_size 16 --n_epochs 20 --lr 1e-2
#python main_skeleton_classifier.py --model ctr_gcn --output_path CTR_GCN-20-1e-3 --batch_size 16 --n_epochs 20 --lr 1e-3
#python main_skeleton_classifier.py --model ctr_gcn --output_path CTR_GCN-20-1e-4 --batch_size 16 --n_epochs 20 --lr 1e-4
#python main_skeleton_classifier.py --model mlp --output_path MLP-20-1e-3 --batch_size 32 --n_epochs 20 --lr 1e-3
#python main_skeleton_classifier.py --model mlp_trans_enc --output_path MLPTransformerModel-20-1e-4 --batch_size 32 --n_epochs 20 --lr 1e-4