import sys
sys.path.insert(1, '/home/samuel/EFARS/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import ConcatDataset

from models.st_gcn import ST_GCN_18
from models.ctr_gcn import CTR_GCN
from models.mlp import MLP, MLPTransformerModel

from data.mm_fit import MMFit, MMFitMetaData
from utils.misc import seed_everything
from utils.parser import args
from utils.fitter import PoseSkeletonClassificationFitter, get_config
from utils.metrics import Accuracy

seed_everything(args.seed)

root_path = '/home/samuel/mm-fit'

#TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
#VAL_W_IDs = ['14', '15', '19']
TRAIN_W_IDs = ['01']
VAL_W_IDs = ['14']

train_dataset = []
val_dataset = []

for id in TRAIN_W_IDs + VAL_W_IDs:
    pose_path = f'{root_path}/w{id}/w{id}_pose_3d.npy'
    label_path = f'{root_path}/w{id}/w{id}_labels.csv'
    if id in TRAIN_W_IDs:
        train_dataset.append(MMFit(pose_path, label_path, 8, 5))
    else:
        val_dataset.append(MMFit(pose_path, label_path, 8, 5))

train_dataset = ConcatDataset(train_dataset)
val_dataset = ConcatDataset(val_dataset)

num_train_samples = len(train_dataset)
num_valid_samples = len(val_dataset)

print(num_train_samples, num_valid_samples)
    
if args.model == 'st_gcn':
    net = ST_GCN_18(in_channels=3, num_class=MMFitMetaData.num_classes)
elif args.model == 'ctr_gcn':
    net = CTR_GCN(num_class=MMFitMetaData.num_classes, num_point=MMFitMetaData.num_joints, num_person=1)
elif args.model == 'mlp':
    net = MLP(num_joint=MMFitMetaData.num_joints, num_classes=MMFitMetaData.num_classes)
elif args.model == 'mlp_trans_enc':
    net = MLPTransformerModel(128, MMFitMetaData.num_joints, MMFitMetaData.num_classes)

num_gpus = torch.cuda.device_count()
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size if num_gpus == 0 else args.batch_size * num_gpus,
    sampler=SequentialSampler(train_dataset),
    pin_memory=False,
    drop_last=True,
    num_workers=args.num_workers,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=args.batch_size if num_gpus == 0 else args.batch_size * num_gpus,
    num_workers=args.num_workers,
    shuffle=False,
    sampler=SequentialSampler(val_dataset),
    pin_memory=False,
)

cfg = get_config(args, nn.CrossEntropyLoss(), Accuracy(), train_loader)
fitter = PoseSkeletonClassificationFitter(net, cfg)
fitter.fit(train_loader, val_loader)