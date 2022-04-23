import sys
sys.path.insert(1, '/home/samuel/EFARS/')

import torch
import torch.nn as nn
import os
from glob import glob
from datetime import datetime
import time

from utils.misc import AverageMeter


class Fitter:
    def __init__(self, model, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'checkpoints/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_loss = 10**5

        self.model = model
        if self.config.cuda:
            self.model = self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = config.scheduler(self.optimizer, **config.scheduler_params)
        self.criterion = config.criterion
        self.metrics = config.metrics
        if self.config.cuda:
            self.criterion = self.criterion.cuda()

        self.log(f'Fitter prepared. Model: {self.model.__class__.__name__}, ' + \
                 f'Criterion: {self.criterion.__class__.__name__}, ' + \
                 f'Metrics: {self.metrics.__class__.__name__}, ' + \
                 f'Learning Rate: {self.config.lr}, ' + \
                 f'Batch Size: {self.config.batch_size}, ' + \
                 f'Epochs: {self.config.n_epochs}')

        if self.config.cuda and torch.cuda.device_count() > 1:
            print(f"DataParallel is used. Number of GPUs: {torch.cuda.device_count()}")
            self.model = nn.DataParallel(self.model)

    def fit(self, train_loader, validation_loader):
        for _ in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            loss, metrics = self.train(train_loader)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, Loss: {loss.avg:.8f}, ' + \
                     f'Metrics: {metrics.avg}, Time: {(time.time() - t):.5f}')
            
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            loss, metrics = self.validate(validation_loader)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, Loss: {loss.avg:.8f}, ' + \
                     f'Metrics: {metrics.avg}, Time: {(time.time() - t):.5f}')

            if loss.avg < self.best_loss:
                self.best_loss = loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=loss.avg)

            self.epoch += 1

    def train(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        summary_metrics = AverageMeter()
        t = time.time()
        for step, data in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(f'Train Step {step}/{len(train_loader)}, Loss: {summary_loss.avg:.8f}, ' + \
                          f'Metrics: {summary_metrics.avg}, Time: {(time.time() - t):.5f}', end='\r')

            loss, metrics = self.compute_loss_and_metrics(data, train=True)
            summary_loss.update(loss.detach().item(), self.config.batch_size // torch.cuda.device_count())
            summary_metrics.update(metrics.cpu().detach().numpy(), self.config.batch_size // torch.cuda.device_count())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer) # native fp16
            
            if self.config.step_scheduler:
                self.scheduler.step()
            
            self.scaler.update() #native fp16

        return summary_loss, summary_metrics

    def validate(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_metrics = AverageMeter()
        t = time.time()
        for step, data in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(f'Val Step {step}/{len(val_loader)}, Loss: {summary_loss.avg:.8f}, ' + \
                          f'Metrics: {summary_metrics.avg}, Time: {(time.time() - t):.5f}', end='\r')

            with torch.no_grad():
                loss, metrics = self.compute_loss_and_metrics(data, train=False)

            summary_loss.update(loss.detach().item(), self.config.batch_size // torch.cuda.device_count())
            summary_metrics.update(metrics.cpu().detach().numpy(), self.config.batch_size // torch.cuda.device_count())

        return summary_loss, summary_metrics

    def test(self, test_loader):
        self.log('Finished training. Start testing...')
        return self.validate(test_loader)

    def compute_loss_and_metrics(self, data, train):
        return 0, 0

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path, only_model=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not only_model:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_loss = checkpoint['best_summary_loss']
            self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

    def _compute_loss_and_metrics(self, input, gt):
        if self.config.cuda:
            input = input.cuda()
            gt = gt.cuda()
        with torch.cuda.amp.autocast():
            preds = self.model(input)
            loss = self.criterion(preds, gt)
            metrics = self.metrics(preds, gt)
        return loss, metrics

class PoseEstimation2DFitter(Fitter):
    def __init__(self, model, config):
        super().__init__(model, config)
    
    def compute_loss_and_metrics(self, data, train):
        imgs, hmaps, _ = data
        imgs = imgs.float()
        hmaps = hmaps.float()
        if self.config.cuda:
            imgs = imgs.cuda()
            hmaps = hmaps.cuda()
        with torch.cuda.amp.autocast():
            preds = self.model(imgs)
            if preds.shape[1] != hmaps.shape[1] and train:
                hmaps_sum = hmaps.sum(dim=1).unsqueeze(1)
                hmaps_minus = torch.clamp(1 - hmaps_sum, 0, 1)
                hmaps = torch.cat((hmaps, hmaps_minus), dim=1)
            loss = self.criterion(preds, hmaps)
            metrics = self.metrics(preds, hmaps)
        return loss, metrics

class Pose2Dto3DFitter(Fitter):
    def __init__(self, model, config):
        super().__init__(model, config)
    
    def compute_loss_and_metrics(self, data, train):
        pos2ds, pos3ds, _, _, _ = data
        pos2ds = pos2ds.float()
        pos3ds = pos3ds.float()
        return self._compute_loss_and_metrics(pos2ds, pos3ds)

class PoseVideoClassificationFitter(Fitter):
    def __init__(self, model, config):
        super().__init__(model, config)
    
    def compute_loss_and_metrics(self, data, train):
        imgs, _, labels = data
        imgs = imgs.permute((0, 4, 1, 2, 3))
        imgs = imgs.float()
        labels = labels.long()
        return self._compute_loss_and_metrics(imgs, labels)

class PoseSkeletonClassificationFitter(Fitter):
    def __init__(self, model, config):
        super().__init__(model, config)
    
    def compute_loss_and_metrics(self, data, train):
        poses, labels, _ = data
        poses = poses.float()
        labels = labels.long()
        return self._compute_loss_and_metrics(poses, labels)

class FitterDefaultConfig:
    num_workers = 8
    batch_size = 32
    n_epochs = 60
    cuda = True

    folder = 'test'
    lr = 0.01

    verbose = True
    verbose_step = 1

    criterion = nn.MSELoss()
    metrics = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(patience=5, factor=0.1)
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

def get_config(args, criterion, metrics, data_loader):
    cfg = FitterDefaultConfig
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size * torch.cuda.device_count()
    cfg.n_epochs = args.n_epochs
    cfg.folder = args.output_path
    cfg.lr = args.lr
    cfg.criterion = criterion
    cfg.metrics = metrics
    if args.scheduler == 'onecycle':
        cfg.scheduler = torch.optim.lr_scheduler.OneCycleLR
        cfg.step_scheduler = True
        cfg.validation_scheduler = False
        cfg.scheduler_params = dict(
            max_lr=cfg.lr,
            epochs=cfg.n_epochs,
            steps_per_epoch=int(len(data_loader)),
            pct_start=args.pct_start,
            anneal_strategy=args.anneal_strategy, 
            final_div_factor=args.final_div_factor
        )
    elif args.scheduler == 'plateau':
        cfg.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        cfg.step_scheduler = False
        cfg.validation_scheduler = True
        cfg.scheduler_params = dict(
            patience=args.patience,
            factor=args.factor
        )
    else:
        raise NotImplementedError
    return cfg