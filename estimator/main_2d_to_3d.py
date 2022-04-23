import sys
sys.path.insert(1, '/home/samuel/EFARS/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import random
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from models.sem_gcn import SemGCN
from estimator.models.mlp import MLP
from data.human36m import Human36M2DTo3DDataset, Human36MMetadata
from utils.misc import seed_everything
from utils.graph import adj_mx_from_edges
from utils.parser import args
from utils.fitter import Pose2Dto3DFitter, get_config
from utils.metrics import MPJPE_PMPJPE

seed_everything(args.seed)

root_path = '/home/samuel/h36m'
img_path = root_path + '/imgs'
pos2d_path = root_path + '/pos2d'
pos3d_path = root_path + '/pos3d'

img_fns = glob(img_path+'/S[0-9]_*.jpg')
split = int(0.8*len(img_fns))
random.shuffle(img_fns)
train_fns = img_fns[:50000]
val_fns = img_fns[50000:60000]
test_fns = img_fns[60000:70000]

train_dataset = Human36M2DTo3DDataset(train_fns, pos2d_path, pos3d_path)
val_dataset = Human36M2DTo3DDataset(val_fns, pos2d_path, pos3d_path)
test_dataset = Human36M2DTo3DDataset(test_fns, pos2d_path, pos3d_path)

if args.model == 'sem_gcn':
    adj = adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False)
    net = SemGCN(adj=adj, num_layers=4, hid_dim=args.hid_dim)
elif args.model == 'mlp':
    net = MLP(num_joint=Human36MMetadata.num_classes)

num_gpus = torch.cuda.device_count()
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size if num_gpus == 0 else args.batch_size * num_gpus,
    num_workers=args.num_workers,
    sampler=SequentialSampler(train_dataset),
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

cfg = get_config(args, nn.MSELoss(), MPJPE_PMPJPE(), train_loader)
fitter = Pose2Dto3DFitter(net, cfg)

if args.test:
    fitter.test(test_loader, args.checkpoint)
else:
    if args.checkpoint != None:
        fitter.load(args.checkpoint)
    fitter.fit(train_loader, val_loader)
    if args.test_after_train:
        fitter.test(test_loader, f'{fitter.base_dir}/last-checkpoint.bin')