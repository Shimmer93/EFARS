import sys
sys.path.insert(1, '/home/samuel/EFARS/')

import torch
import torch.nn as nn

from torch.utils.data import random_split, ConcatDataset
from torchvision.datasets import HMDB51
import torchvision.transforms as T

from models.cnn_lstm import CNNLSTM
from models.cnn_3d import ShuffleNetV2, MobileNetV2
from models.timesformer.vit import TimeSformer

from data.hmdb51 import HMDB51Metadata
from utils.misc import seed_everything
from utils.parser import args
from utils.fitter import PoseVideoClassificationFitter, get_config
from utils.metrics import Accuracy
from utils.data import TrainDataLoader, ValDataLoader, TestDataLoader
if args.gpus != None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

seed_everything(args.seed)

root_path = '/home/samuel/hmdb51/videos'
annot_path = '/home/samuel/hmdb51/annots'

transforms = T.Compose([
    T.Lambda(lambda tensor: tensor.permute(0, 3, 1, 2)),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=HMDB51Metadata.mean, std=HMDB51Metadata.std),
    T.Resize((128,171)),
    T.RandomCrop((args.crop_size,args.crop_size)),
    T.Lambda(lambda tensor: tensor.permute(0, 2, 3, 1))
])

total_dataset = []
for i in [1, 2, 3]:
    total_dataset.append(HMDB51(root=root_path, annotation_path=annot_path, frames_per_clip=args.seq_len, transform=transforms,
                       step_between_clips=args.frame_interval, fold=i, train=True, num_workers=args.num_workers))
total_dataset = ConcatDataset(total_dataset)

num_total_samples = len(total_dataset)
num_train_samples = int(0.8 * num_total_samples)
num_val_samples = int(0.1 * num_total_samples)
num_test_samples = num_total_samples - num_train_samples - num_val_samples

train_dataset, val_dataset, test_dataset = random_split(total_dataset, [num_train_samples, num_val_samples, num_test_samples])

print(num_train_samples, num_val_samples, num_test_samples)
    
if args.model == 'mobilenetv2':
    net = MobileNetV2(num_classes=HMDB51Metadata.num_classes, sample_size=args.crop_size)
elif args.model == 'shufflenetv2':
    net = ShuffleNetV2(num_classes=HMDB51Metadata.num_classes, sample_size=args.crop_size)
elif args.model == 'cnnlstm':
    net = CNNLSTM(num_classes=HMDB51Metadata.num_classes)
elif args.model == 'timesformer':
    net = TimeSformer(img_size=args.crop_size, num_classes=HMDB51Metadata.num_classes, num_frames=args.seq_len)

num_gpus = torch.cuda.device_count()
train_loader = TrainDataLoader(train_dataset, args.batch_size, num_gpus, args.num_workers)
val_loader = ValDataLoader(val_dataset, args.batch_size, num_gpus, args.num_workers)
test_loader = TestDataLoader(test_dataset, args.batch_size, num_gpus, args.num_worker)

cfg = get_config(args, nn.CrossEntropyLoss(), Accuracy(), train_loader)
fitter = PoseVideoClassificationFitter(net, cfg)

if args.test:
    fitter.test(test_loader, args.checkpoint)
else:
    if args.checkpoint != None:
        fitter.load(args.checkpoint)
    fitter.fit(train_loader, val_loader)
    if args.test_after_train:
        fitter.test(test_loader, f'{fitter.base_dir}/last-checkpoint.bin')