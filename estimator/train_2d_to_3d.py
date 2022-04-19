import sys
sys.path.insert(1, '/home/samuel/EFARS/')

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
from models.pose2mesh import PoseNet
#from models.simple_graph import GCN
from data.human36m import Human36M2DTo3DDataset, Human36MMetadata
from utils.misc import AverageMeter, seed_everything
from utils.graph import adj_mx_from_edges
from utils.transform import do_2d_to_3d_transforms

from utils.parser import args

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

train_dataset = Human36M2DTo3DDataset(train_fns, pos2d_path, pos3d_path, transforms=do_2d_to_3d_transforms)
val_dataset = Human36M2DTo3DDataset(val_fns, pos2d_path, pos3d_path, transforms=do_2d_to_3d_transforms)

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'/home/samuel/EFARS/estimator/checkpoints/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model, device_ids=[0,1])
        self.device = device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = nn.MSELoss().cuda()
        self.log(f'Fitter prepared. Device is {self.device}')
        
        # self.iters_to_accumulate = 4 # gradient accumulation

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            #self.log(f'[RESULT]: Train. Epoch: {self.epoch}, accuracy: {val_loss.avg:.8f}, time: {(time.time() - t):.5f}')
 
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            #self.log(f'[RESULT]: Val. Epoch: {self.epoch}, accuracy: {val_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        mse_loss = AverageMeter()
        t = time.time()
        for step, (pos2ds, pos3ds, _, _, _) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'mse_loss: {mse_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            with torch.no_grad():
                pos2ds = pos2ds.cuda().float()
                pos3ds = pos3ds.cuda().float()
                batch_size = pos2ds.shape[0]
                 
                with torch.cuda.amp.autocast():
                    preds = self.model(pos2ds)
                    loss = self.criterion(preds,pos3ds)

            mse_loss.update(loss.detach().item(), batch_size)
            #self.scaler.scale(loss).backward()

        return mse_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        mse_loss = AverageMeter()
        t = time.time()
        for step, (pos2ds, pos3ds, _, _, _) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'mse_loss: {mse_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            pos2ds = pos2ds.cuda().float()
            pos3ds = pos3ds.cuda().float()
            batch_size = pos2ds.shape[0]
            
            with torch.cuda.amp.autocast():
                preds = self.model(pos2ds)
                loss = self.criterion(preds,pos3ds)

            mse_loss.update(loss.detach().item(), batch_size)
            self.scaler.scale(loss).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # loss = loss / self.iters_to_accumulate # gradient accumulation
            
            #mse_loss.update(loss.detach().item(), batch_size)
            
            #self.optimizer.step()
            self.scaler.step(self.optimizer) # native fp16
            
            if self.config.step_scheduler:
                self.scheduler.step()
            
            self.scaler.update() #native fp16
                
                
#             if (step+1) % self.iters_to_accumulate == 0: # gradient accumulation

#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#                 if self.config.step_scheduler:
#                     self.scheduler.step()

        return mse_loss

    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            #'amp': amp.state_dict() # apex
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
            
            
class TrainGlobalConfig:
    num_workers = args.num_workers
    batch_size = args.batch_size * torch.cuda.device_count()
    n_epochs = args.n_epochs

    folder = args.output_path
    lr = args.max_lr

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau #OneCycleLR #ReduceLROnPlateau
    scheduler_params = dict(
        patience=5,
        factor=0.1,
        #max_lr=args.max_lr,
        ##total_steps = len(train_dataset) // 4 * n_epochs, # gradient accumulation
        #epochs=n_epochs,
        #steps_per_epoch=int(len(train_dataset) / batch_size),
        #pct_start=args.pct_start,
        #anneal_strategy=args.anneal_strategy, 
        #final_div_factor=args.final_div_factor
    )
    
if args.model == 'sem_gcn':
    #edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, 16)), Human36MMetadata.parents2)))
    net = SemGCN(adj=adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False), num_layers=4, hid_dim=128).cuda()
elif args.model == 'gcn':
    #edge_index = adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False)
    edges = torch.tensor(Human36MMetadata.skeleton_edges, dtype=torch.long).T
    #net = GCN(edge_index=edges.cuda(), hidden_channels=128).cuda()
elif args.model == 'pose2mesh':
    net = PoseNet(num_joint=17).cuda()

def run_training():
    device = torch.device('cuda')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=SequentialSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(val_dataset),
        pin_memory=False,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    #fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)
    
    
run_training()