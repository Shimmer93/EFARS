import sys
sys.path.insert(1, '/home/samuel/EFARS/')

import torch
import torch.nn as nn
from glob import glob
import random

from models.sem_gcn import SemGCN
from models.mlp import MLP
from models.video3d import VideoPose3DNonTemporal

from data.human36m import Human36M2DTo3DDataset, Human36MMetadata
from utils.misc import seed_everything
from utils.graph import adj_mx_from_edges
from utils.parser import args
from utils.fitter import Pose2Dto3DFitter, get_config
from utils.metrics import MPJPE_PMPJPE_NMPJPE
from utils.data import TrainDataLoader, ValDataLoader, TestDataLoader
if args.gpus != None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

seed_everything(args.seed)

root_path = '/home/samuel/h36m'
img_path = root_path + '/imgs'
pos2d_path = root_path + '/pos2d'
pos3d_path = root_path + '/pos3d'

img_fns = glob(img_path+'/S[0-9]_*.jpg') # Errors in S11
random.shuffle(img_fns)
train_fns = img_fns[:500]
val_fns = img_fns[500:600]
test_fns = img_fns[600:700]

train_dataset = Human36M2DTo3DDataset(train_fns, pos2d_path, pos3d_path)
val_dataset = Human36M2DTo3DDataset(val_fns, pos2d_path, pos3d_path)
test_dataset = Human36M2DTo3DDataset(test_fns, pos2d_path, pos3d_path)

if args.model == 'sem_gcn':
    adj = adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False)
    net = SemGCN(adj=adj, num_layers=4, hid_dim=args.hid_dim)
elif args.model == 'mlp':
    net = MLP(num_joint=Human36MMetadata.num_classes)
elif args.model == 'videopose3d':
    net = VideoPose3DNonTemporal(Human36MMetadata.num_joints, 2, Human36MMetadata.num_joints, [1,1,1,1,1])

num_gpus = torch.cuda.device_count()
train_loader = TrainDataLoader(train_dataset, args.batch_size, num_gpus, args.num_workers)
val_loader = ValDataLoader(val_dataset, args.batch_size, num_gpus, args.num_workers)
test_loader = TestDataLoader(test_dataset, args.batch_size, num_gpus, args.num_workers)

cfg = get_config(args, nn.MSELoss(), MPJPE_PMPJPE_NMPJPE(), train_loader)
fitter = Pose2Dto3DFitter(net, cfg)

if args.test:
    fitter.test(test_loader, args.checkpoint)
else:
    if args.checkpoint != None:
        fitter.load(args.checkpoint)
    fitter.fit(train_loader, val_loader)
    if args.test_after_train:
        fitter.test(test_loader, f'{fitter.base_dir}/last-checkpoint.bin')