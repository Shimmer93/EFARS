import sys
sys.path.insert(1, '/home/samuel/EFARS/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from glob import glob
import random
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from models.attention import GCNTransformerModel, PureTransformerModel
from models.video3d import TemporalModelOptimized1f

from data.human36m import Human36M2DTo3DTemporalDataset, Human36MMetadata
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
train_fns = img_fns[:1000]
val_fns = img_fns[1000:1200]

train_dataset = Human36M2DTo3DTemporalDataset(train_fns, pos2d_path, pos3d_path, length=8)
val_dataset = Human36M2DTo3DTemporalDataset(val_fns, pos2d_path, pos3d_path, length=8)
    
if args.model == 'gcn_trans_enc':
    adj = adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False)
    net = GCNTransformerModel(adj=adj, num_layers=4, hid_dim=args.hid_dim)
elif args.model == 'trans_enc':
    net = PureTransformerModel(args.hid_dim)
elif args.model == 'videopose3d':
    net = TemporalModelOptimized1f(17, 2, 17, [1,1,1,1,1])

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

cfg = get_config(args, nn.MSELoss(), MPJPE_PMPJPE(), train_loader)
fitter = Pose2Dto3DFitter(net, cfg)
fitter.fit(train_loader, val_loader)