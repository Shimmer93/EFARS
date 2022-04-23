import sys
sys.path.insert(1, '/home/samuel/EFARS/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import numpy as np
from glob import glob
import random
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import albumentations as A

from models.openpose import OpenPose
from models.unipose import UniPose
from data.human36m import Human36M2DPoseDataset, Human36MMetadata
from utils.misc import seed_everything, normalize, get_random_crop_positions_with_pos2d
from utils.parser import args
from utils.fitter import PoseEstimation2DFitter, get_config
from utils.metrics import MPCK_MPCKh

seed_everything(args.seed)

root_path = '/home/samuel/h36m'
img_path = root_path + '/imgs'
pos2d_path = root_path + '/pos2d'

img_fns = glob(img_path+'/*.jpg')
split = int(0.8*len(img_fns))
random.shuffle(img_fns)
train_fns = img_fns[:100]
val_fns = img_fns[100:120]

def train_transforms(img, pos2d):
    #x_min, y_min, x_max, y_max = get_random_crop_positions_with_pos2d(img, pos2d, kwargs['crop_size'])
    transforms = A.Compose([
        #A.Crop(x_min, y_min, x_max, y_max, p=0.5),
        #A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        #A.Blur(blur_limit=3,p=0.2),
        #A.HorizontalFlip(p=0.5),
        A.Resize(height=args.out_size, width=args.out_size, p=1)
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pos2d)
    img = transformed['image']
    img = normalize(img, Human36MMetadata.mean, Human36MMetadata.std)
    pos2d = np.array(transformed['keypoints'])
    return img, pos2d

def val_transforms(img, pos2d):
    transforms = A.Compose([
        A.Resize(height=args.out_size, width=args.out_size, p=1)
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pos2d)
    img = transformed['image']
    img = normalize(img, Human36MMetadata.mean, Human36MMetadata.std)
    pos2d = np.array(transformed['keypoints'])
    return img, pos2d

train_dataset = Human36M2DPoseDataset(train_fns, pos2d_path, transforms=train_transforms, 
    out_size=(args.out_size, args.out_size), mode='E', sigma=args.sigma)
val_dataset = Human36M2DPoseDataset(val_fns, pos2d_path, transforms=val_transforms, 
    out_size=(args.out_size, args.out_size), mode='E', sigma=args.sigma)

if args.model == 'unipose':
    net = UniPose(dataset='human3.6m', num_classes=17)
elif args.model == 'openpose':
    net = OpenPose()

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

cfg = get_config(args, nn.MSELoss(), MPCK_MPCKh(), train_loader)
fitter = PoseEstimation2DFitter(net, cfg)
fitter.fit(train_loader, val_loader)