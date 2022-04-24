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