import sys
sys.path.insert(1, '/home/zpengac/pose/EFARS/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import random
from datetime import datetime
import time
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from models.sem_gcn import SemGCN
from estimator.models.mlp import MLP
from data.human36m import Human36M2DTo3DDataset, Human36M2DTo3DTemporalDataset, Human36MMetadata
from utils.misc import AverageMeter, seed_everything, accuracy
from utils.graph import adj_mx_from_edges

seed_everything(2333)

root_path = '/scratch/PI/cqf/datasets/h36m'
img_path = root_path + '/img'
pos2d_path = root_path + '/pos2d'
pos3d_path = root_path + '/pos3d'

img_fns = glob(img_path+'/*.jpg')
split = int(0.8*len(img_fns))
random.shuffle(img_fns)
test_fns = img_fns[12000:14000]

test_dataset = Human36M2DTo3DDataset(test_fns, pos2d_path, pos3d_path)
#test_dataset = Human36M2DTo3DTemporalDataset(test_fns, pos2d_path, pos3d_path)

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

class Tester:
    def __init__(self, model, device, config):
        self.config = config
        self.base_dir = f'/home/zpengac/pose/EFARS/estimator/checkpoints/{config.folder}'

        assert os.path.exists(self.base_dir)

        self.model = model
        #if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #    self.model = nn.DataParallel(self.model)
        self.device = device
        self.criteron = nn.MSELoss()#.cuda()
        self.numClasses = 17

    def test(self, test_loader):
        self.model.eval()
        mse_loss = AverageMeter()
        mpjpe_meter = AverageMeter()

        for i, (pos2ds, pos3ds) in enumerate(test_loader):
            with torch.no_grad():
                pos2ds = pos2ds.float()#.cuda()
                pos3ds = pos3ds.float()#.cuda()
                batch_size = pos2ds.shape[0]
                
                with torch.cuda.amp.autocast():
                    preds = self.model(pos2ds)
                    loss = self.criteron(preds, pos3ds)
                    mpjpe_value = mpjpe(preds, pos3ds)

            mse_loss.update(loss.detach().item(), batch_size)
            mpjpe_meter.update(mpjpe_value.detach().item(), batch_size)

        print(f'loss: {mse_loss.avg}, MPJPE: {mpjpe_meter.avg}')

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            new_sd[k[7:]] = v
        self.model.load_state_dict(new_sd)
        #self.model.load_state_dict(checkpoint['model_state_dict'])            


class TestGlobalConfig:
    num_workers = 8
    batch_size = 8 #* torch.cuda.device_count()
    folder = 'GCNLSTM-60-1e-2'
    
#net = SemGCN(adj=adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False), num_layers=4, hid_dim=128).cuda()
net = MLP(num_joint=17).cuda()

def run_testing():
    device = torch.device('cpu')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=TestGlobalConfig.batch_size,
        num_workers=TestGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
    )

    tester = Tester(model=net, device=device, config=TestGlobalConfig)
    tester.load(f'{tester.base_dir}/best-checkpoint-009epoch.bin')
    tester.test(test_loader)
    
run_testing()