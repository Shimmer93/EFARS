import sys
sys.path.insert(1, '/home/samuel/EFARS/')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn

from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import random_split, ConcatDataset
from torchvision.datasets import HMDB51
import torchvision.transforms as T

from models.cnn_lstm import CNNLSTM
from models.cnn_3d import ShuffleNetV2, MobileNetV2
from models.timesformer.vit import TimeSformer

from utils.misc import seed_everything
from utils.parser import args
from utils.fitter import PoseVideoClassificationFitter, get_config
from utils.metrics import Accuracy

seed_everything(args.seed)

root_path = '/home/samuel/hmdb51/videos'
annot_path = '/home/samuel/hmdb51/annots'

transforms = T.Compose([
    T.Lambda(lambda tensor: tensor.permute(0, 3, 1, 2)),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    T.Resize((128,171)),
    T.RandomCrop((112,112)),
    T.Lambda(lambda tensor: tensor.permute(0, 2, 3, 1))
])

total_dataset = []
for i in [1]:
    total_dataset.append(HMDB51(root=root_path, annotation_path=annot_path, frames_per_clip=8, transform=transforms,
                       step_between_clips=5, fold=i, train=True, num_workers=args.num_workers))
total_dataset = ConcatDataset(total_dataset)

num_total_samples = len(total_dataset)
num_train_samples = int(0.8 * num_total_samples)
num_val_samples = num_total_samples - num_train_samples

#train_dataset, val_dataset = random_split(total_dataset, [num_train_samples, num_val_samples])
train_dataset, val_dataset, test_dataset = random_split(total_dataset, [200, 50, num_total_samples-250])

print(num_total_samples, num_train_samples, num_val_samples)
    
if args.model == 'mobilenetv2':
    net = MobileNetV2(num_classes=26, sample_size=112)
elif args.model == 'shufflenetv2':
    net = ShuffleNetV2(num_classes=26, sample_size=112)
elif args.model == 'cnnlstm':
    net = CNNLSTM(num_classes=26)
elif args.model == 'timesformer':
    net = TimeSformer(img_size=112, num_classes=26, num_frames=8)

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

cfg = get_config(args, nn.CrossEntropyLoss(), Accuracy(), train_loader)
fitter = PoseVideoClassificationFitter(net, cfg)
fitter.fit(train_loader, val_loader)