import sys
sys.path.insert(1, '/home/samuel/EFARS/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
import tqdm
import random
from datetime import datetime
import time
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import random_split, ConcatDataset
import torchvision.transforms as T

#from models.torchvision_models import ResNet18, ResNet50, MobileNetV3Small
from models.st_gcn import ST_GCN_18
from models.ctr_gcn import CTR_GCN
from models.mlp import MLP, MLPTransformerModel

from data.mm_fit import MMFit, MMFitMetaData
from utils.misc import AverageMeter, seed_everything
from utils.parser import args

seed_everything(args.seed)

root_path = '/home/samuel/mm-fit'

TRAIN_W_IDs = ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18']
VAL_W_IDs = ['14', '15', '19']

train_dataset = []
valid_dataset = []

for id in TRAIN_W_IDs + VAL_W_IDs:
    pose_path = f'{root_path}/w{id}/w{id}_pose_3d.npy'
    label_path = f'{root_path}/w{id}/w{id}_labels.csv'
    if id in TRAIN_W_IDs:
        train_dataset.append(MMFit(pose_path, label_path, 8, 5))
    else:
        valid_dataset.append(MMFit(pose_path, label_path, 8, 5))

train_dataset = ConcatDataset(train_dataset)
valid_dataset = ConcatDataset(valid_dataset)

num_train_samples = len(train_dataset)
num_valid_samples = len(valid_dataset)

print(num_train_samples, num_valid_samples)

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'/home/samuel/EFARS/classifier/checkpoints/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #self.model = nn.DataParallel(self.model)
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = nn.CrossEntropyLoss().cuda()#ContrastiveLoss()
        #self.metric = torch.dist#nn.CosineSimilarity()
        self.log(f'Fitter prepared. Device is {self.device}')
        
        # self.iters_to_accumulate = 4 # gradient accumulation

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, accuracy = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, ce_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, accuracy: {accuracy.avg:.8f}, time: {(time.time() - t):.5f}')
 
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, accuracy = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, ce_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, accuracy: {accuracy.avg:.8f}, time: {(time.time() - t):.5f}')
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
        ce_loss = AverageMeter()
        accuracy = AverageMeter()
        t = time.time()
        for step, (poses, labels, _) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'ce_loss: {ce_loss.avg:.8f}, ' + \
                        f'accuracy: {accuracy.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            with torch.no_grad():
                poses = poses.cuda().float()
                if args.model == 'mlp':
                    poses = poses.squeeze().transpose(1, 2)
                elif args.model == 'mlp_trans_enc':
                    poses = poses.squeeze().permute(0, 2, 3, 1)
                #skeletons = skeletons.cuda().float()
                labels = labels.cuda()
                batch_size = poses.shape[0]
                
                with torch.cuda.amp.autocast():
                    preds = self.model(poses)
                    loss = self.criterion(preds,labels)

            ce_loss.update(loss.detach().item(), batch_size)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            accuracy.update(acc.detach().item(), batch_size)
            #self.scaler.scale(loss).backward()

        return ce_loss, accuracy

    def train_one_epoch(self, train_loader):
        self.model.train()
        ce_loss = AverageMeter()
        accuracy = AverageMeter()
        t = time.time()
        for step, (poses, labels, _) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'ce_loss: {ce_loss.avg:.8f}, ' + \
                        f'accuracy: {accuracy.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            poses = poses.cuda().float()
            if args.model == 'mlp':
                poses = poses.squeeze().transpose(1, 2)
            elif args.model == 'mlp_trans_enc':
                poses = poses.squeeze().permute(0, 2, 3, 1)
            #skeletons = skeletons.cuda().float()
            labels = labels.cuda()
            batch_size = poses.shape[0]
            
            with torch.cuda.amp.autocast():
                preds = self.model(poses)
                loss = self.criterion(preds,labels)

            ce_loss.update(loss.detach().item(), batch_size)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            accuracy.update(acc.detach().item(), batch_size)

            self.scaler.scale(loss).backward()
            # loss = loss / self.iters_to_accumulate # gradient accumulation
            
            #ce_loss.update(loss.detach().item(), batch_size)
            
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

        return ce_loss, accuracy

    
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
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #self.best_summary_loss = checkpoint['best_summary_loss']
        #self.epoch = checkpoint['epoch'] + 1
        
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
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR #OneCycleLR ReduceLROnPlateau
    scheduler_params = dict(
        #patience = 5,
        #factor = 0.1,
        max_lr=args.max_lr,
        #total_steps = len(train_dataset) // 4 * n_epochs, # gradient accumulation
        epochs=n_epochs,
        steps_per_epoch=int(len(train_dataset) / batch_size),
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy, 
        final_div_factor=args.final_div_factor
    )
    
if args.model == 'st_gcn':
    net = ST_GCN_18(in_channels=3, num_class=MMFitMetaData.num_classes).cuda()
elif args.model == 'ctr_gcn':
    net = CTR_GCN(num_class=MMFitMetaData.num_classes, num_point=MMFitMetaData.num_joints, num_person=1).cuda()
elif args.model == 'mlp':
    net = MLP(num_joint=MMFitMetaData.num_joints, num_classes=MMFitMetaData.num_classes).cuda()
elif args.model == 'mlp_trans_enc':
    net = MLPTransformerModel(128, MMFitMetaData.num_joints, MMFitMetaData.num_classes, 8).cuda()


def run_training():
    device = torch.device('cuda')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.load('/home/samuel/EFARS/classifier/checkpoints/ST-GCN-20-1e-2/best-checkpoint-003epoch.bin')
    fitter.fit(train_loader, val_loader)
    
    
run_training()