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

TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
VAL_W_IDs = ['14', '15', '19']
TEST_W_IDs = ['00', '05', '12', '13', '20']

train_dataset = []
val_dataset = []
test_dataset = []

for id in TRAIN_W_IDs + VAL_W_IDs + TEST_W_IDs:
    pose_path = f'{root_path}/w{id}/w{id}_pose_3d.npy'
    label_path = f'{root_path}/w{id}/w{id}_labels.csv'
    dataset = MMFit(pose_path, label_path, args.seq_len, args.frame_interval)
    if id in TRAIN_W_IDs:
        train_dataset.append(dataset)
    elif id in VAL_W_IDs:
        val_dataset.append(dataset)
    else:
        test_dataset.append(dataset)

train_dataset = ConcatDataset(train_dataset)
val_dataset = ConcatDataset(val_dataset)
test_dataset = ConcatDataset(test_dataset)

num_train_samples = len(train_dataset)
num_val_samples = len(val_dataset)
num_test_samples = len(test_dataset)

print(num_train_samples, num_val_samples, num_test_samples)
    
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
    num_workers=args.num_workers,
    sampler=RandomSampler(train_dataset),
    drop_last=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=args.batch_size if num_gpus == 0 else args.batch_size * num_gpus,
    num_workers=args.num_workers,
    sampler=SequentialSampler(val_dataset),
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=args.batch_size if num_gpus == 0 else args.batch_size * num_gpus,
    num_workers=args.num_workers,
    sampler=SequentialSampler(test_dataset),
    shuffle=False,
)

cfg = get_config(args, nn.CrossEntropyLoss(), Accuracy(), train_loader)
fitter = PoseSkeletonClassificationFitter(net, cfg)

if args.test:
    fitter.test(test_loader, args.checkpoint)
else:
    if args.checkpoint != None:
        fitter.load(args.checkpoint)
    fitter.fit(train_loader, val_loader)
    if args.test_after_train:
        fitter.test(test_loader, f'{fitter.base_dir}/last-checkpoint.bin')