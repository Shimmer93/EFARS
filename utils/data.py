from glob import glob
from tqdm import tqdm
import cdflib
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data import DataLoader
import os
import numpy as np

def TrainDataLoader(train_dataset, batch_size, num_gpus, num_workers):
    return DataLoader(
        train_dataset,
        batch_size = batch_size if num_gpus == 0 else batch_size * num_gpus,
        num_workers = num_workers,
        sampler = RandomSampler(train_dataset),
        drop_last = True,
    )

def ValDataLoader(val_dataset, batch_size, num_gpus, num_workers):
    return DataLoader(
        val_dataset, 
        batch_size = batch_size if num_gpus == 0 else batch_size * num_gpus,
        num_workers = num_workers,
        sampler = SequentialSampler(val_dataset),
        shuffle = False,
    )

def TestDataLoader(test_dataset, batch_size, num_gpus, num_workers):
    return ValDataLoader(test_dataset, batch_size, num_gpus, num_workers)

def h36m_pos2d_preprocess(input_dir, output_dir=None):
    '''
        Convert pos2d data from .cdf to .npy and slice it to make it consistent with image data
    '''
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D2_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,2)
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts)

def h36m_pos3d_preprocess(input_dir, output_dir=None):
    '''
        Convert pos3d data from .cdf to .npy and slice it to make it consistent with image data
    '''
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D3_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,3)
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts)