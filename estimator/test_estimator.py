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

from models.openpose import OpenPose
from models.unipose import UniPose
from data.human36m import Human36M2DPoseDataset
from utils.transform import do_pos2d_train_transforms, do_pos2d_val_transforms
from utils.misc import AverageMeter, seed_everything, accuracy

seed_everything(2333)

root_path = '/scratch/PI/cqf/datasets/h36m'
img_path = root_path + '/img'
pos2d_path = root_path + '/pos2d'

img_fns = glob(img_path+'/*.jpg')
split = int(0.8*len(img_fns))
random.shuffle(img_fns)
test_fns = img_fns[12000:14000]

test_dataset = Human36M2DPoseDataset(test_fns, pos2d_path, transforms=do_pos2d_train_transforms, mode='E', sigma=1)

class Tester:
    def __init__(self, model, device, config):
        self.config = config
        self.base_dir = f'/home/zpengac/pose/EFARS/estimator/checkpoints/{config.folder}'

        assert os.path.exists(self.base_dir)

        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
        self.device = device
        self.criteron = nn.MSELoss().cuda()
        self.numClasses = 17

    def test(self, test_loader):
        self.model.eval()
        mse_loss = AverageMeter()
        #mAcc = AverageMeter()
        #mPCK = AverageMeter()
        #mPCKh = AverageMeter()

        AP    = np.zeros(self.numClasses)
        PCK   = np.zeros(self.numClasses)
        PCKh  = np.zeros(self.numClasses)
        count = np.zeros(self.numClasses)

        for imgs, hmaps, skeletons in tqdm(test_loader):
            with torch.no_grad():
                imgs = imgs.float().cuda()
                hmaps = hmaps.float().cuda()
                batch_size = imgs.shape[0]
                
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss = self.criteron(preds, hmaps)
                    acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(preds.cpu().detach().numpy(), hmaps.cpu().detach().numpy(), 0.2, 0.5)

            mse_loss.update(loss.detach().item(), batch_size)

            for j in range(1,self.numClasses):
                if visible[j] == 1:
                    AP[j]     = (AP[j]  *count[j] + acc[j])      / (count[j] + 1)
                    PCK[j]    = (PCK[j] *count[j] + acc_PCK[j])  / (count[j] + 1)
                    PCKh[j]   = (PCKh[j]*count[j] + acc_PCKh[j]) / (count[j] + 1)
                    count[j] += 1

        mAP     =   AP.mean()
        mPCK    =  PCK.mean()
        mPCKh   = PCKh.mean()

            #mAcc.update(acc, batch_size)
            #mPCK.update(acc_PCK, batch_size)
            #mPCKh.update(acc_PCKh, batch_size)

        #mmAcc = np.mean(mAcc.avg).item()
        #mmPCK = np.mean(mPCK.avg).item()
        #mmPCKh = np.mean(mPCKh.avg).item()

        print(f'loss: {mse_loss.avg}, Acc: {mAP}, PCK: {mPCK}, PCKh: {mPCKh}')

    def load(self, path):
        checkpoint = torch.load(path)
        #from collections import OrderedDict
        #new_sd = OrderedDict()
        #for k, v in checkpoint['model_state_dict'].items():
        #    new_sd[k[7:]] = v
        #self.model.load_state_dict(new_sd)
        self.model.load_state_dict(checkpoint['model_state_dict'])            


class TestGlobalConfig:
    num_workers = 8
    batch_size = 1 * torch.cuda.device_count()

    folder = 'UniPose-80-1e-2-sigma1'
    
net = UniPose(dataset='human3.6m',num_classes=17).cuda()
#net = OpenPose().cuda()


def run_testing():
    device = torch.device('cuda')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=TestGlobalConfig.batch_size,
        num_workers=TestGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(test_dataset),
        pin_memory=False,
    )

    tester = Tester(model=net, device=device, config=TestGlobalConfig)
    tester.load(f'{tester.base_dir}/best-checkpoint-010epoch.bin')
    tester.test(test_loader)
    
run_testing()