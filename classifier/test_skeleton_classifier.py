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
from data.mm_fit import MMFit
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

#from models.torchvision_models import ResNet18, ResNet50, MobileNetV3Small
from models.st_gcn import ST_GCN_18
from models.ctr_gcn import CTR_GCN
from models.mlp import MLP, MLPTransformerModel

from data.human36m import Human36M2DTemporalDataset, Human36MMetadata
from utils.misc import AverageMeter, seed_everything
#from utils.parser import args

seed_everything(2333)

root_path = '/home/samuel/mm-fit'

# TODO: transforms

#train_dataset = Human36M2DTemporalDataset(train_fns, pos2d_path, transforms=do_pos2d_train_transforms, mode='C')
#val_dataset = Human36M2DTemporalDataset(val_fns, pos2d_path, transforms=do_pos2d_val_transforms, mode='C')

TEST_W_IDs = ['00', '05', '12', '13', '20']

test_dataset = []

for id in TEST_W_IDs:
    pose_path = f'{root_path}/w{id}/w{id}_pose_3d.npy'
    label_path = f'{root_path}/w{id}/w{id}_labels.csv'
    test_dataset.append(MMFit(pose_path, label_path, 1, 5))

test_dataset = ConcatDataset(test_dataset)

CLASSES = MMFit(f'{root_path}/w00/w00_pose_3d.npy', f'{root_path}/w00/w00_labels.csv', 8).ACTIONS.keys()

num_test_samples = len(test_dataset)

print(num_test_samples)

class Fitter:
    def __init__(self, model, device, config):
        self.config = config

        #self.base_dir = f'/home/samuel/EFARS/classifier/checkpoints/{config.folder}'
        #if not os.path.exists(self.base_dir):
        #    os.makedirs(self.base_dir)
        #
        #self.log_path = f'{self.base_dir}/log.txt'

        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #self.model = nn.DataParallel(self.model)
        self.device = device

        self.criterion = nn.CrossEntropyLoss().cuda()#ContrastiveLoss()
        #self.metric = torch.dist#nn.CosineSimilarity()
        print(f'Fitter prepared. Device is {self.device}')
        
        # self.iters_to_accumulate = 4 # gradient accumulation

    def fit(self, test_loader):

        t = time.time()
        summary_loss, accuracy = self.test(test_loader)

        print(f'[RESULT]: Test. ce_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
        print(f'[RESULT]: Test. accuracy: {accuracy.avg:.8f}, time: {(time.time() - t):.5f}')

    def test(self, test_loader):
        self.model.eval()
        ce_loss = AverageMeter()
        accuracy = AverageMeter()
        t = time.time()
        prs = []
        gts = []
        for step, (poses, labels, _) in enumerate(test_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Test Step {step}/{len(test_loader)}, ' + \
                        f'ce_loss: {ce_loss.avg:.8f}, ' + \
                        f'accuracy: {accuracy.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            with torch.no_grad():
                poses = poses.cuda().float()
                #if args.model == 'mlp':
                poses = poses.squeeze().transpose(1, 2)
                #elif args.model == 'mlp_trans_enc':
                #poses = poses.squeeze().permute(0, 2, 3, 1)
                #skeletons = skeletons.cuda().float()
                labels = labels.cuda()
                batch_size = poses.shape[0]
                
                with torch.cuda.amp.autocast():
                    preds = self.model(poses)
                    loss = self.criterion(preds,labels)

                    prs.extend(preds.argmax(dim=-1).cpu().numpy())
                    gts.extend(labels.cpu().numpy())

            ce_loss.update(loss.detach().item(), batch_size)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            accuracy.update(acc.detach().item(), batch_size)

        cf_matrix = confusion_matrix(gts, prs)
        per_cls_num = cf_matrix.sum(axis=0)
        per_cls_acc = cf_matrix.diagonal()/per_cls_num
        print(per_cls_acc)
        #df_cm = pd.DataFrame((cf_matrix/per_cls_num).transpose(), CLASSES, CLASSES)     # https://sofiadutta.github.io/datascience-ipynbs/pytorch/Image-Classification-using-PyTorch.html
        #plt.figure(figsize = (40,40))
        #sn.set(font_scale=3)
        #sn.heatmap(df_cm, annot=True)
        #plt.xlabel("prediction")
        #plt.ylabel("label (ground truth)")
        #plt.xticks(rotation=45)
        #plt.yticks(rotation=45)
        #plt.savefig("confusion_matrix.png")

        return ce_loss, accuracy

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
            
            
class TestGlobalConfig:
    num_workers = 8
    batch_size = 32 * torch.cuda.device_count()
    
    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------


#if args.model == 'st_gcn':
#net = ST_GCN_18(in_channels=3, num_class=11).cuda()
#elif args.model == 'ctr_gcn':
#    net = CTR_GCN(num_class=11, num_point=17, num_person=1).cuda()
#elif args.model == 'mlp':
net = MLP(num_joint=17, num_classes=11).cuda()
#elif args.model == 'mlp_trans_enc':
#net = MLPTransformerModel(128, 17, 11, 8).cuda()


def run_testing():
    device = torch.device('cuda')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TestGlobalConfig.batch_size,
        sampler=RandomSampler(test_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TestGlobalConfig.num_workers,
    )

    fitter = Fitter(model=net, device=device, config=TestGlobalConfig)
    fitter.load('/home/samuel/EFARS/classifier/checkpoints/MLP-20-1e-3/best-checkpoint-016epoch.bin')
    fitter.fit(test_loader)
    
    
run_testing()