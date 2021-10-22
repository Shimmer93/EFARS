import torch
from models.openpose import openpose
from models.unipose import unipose
from data.human36m import Human36M2DPoseDataset

def seed_everything(seed):
    