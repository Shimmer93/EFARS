import random
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import cdflib

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed):
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True

def normalize(img, mean, std):
    for i in range(img.shape[-1]):
        img[...,i] = (img[...,i] - mean[i]) / std[i]
    return img

def denormalize(img, mean, std):
    for i in range(img.shape[-1]):
        img[...,i] = img[...,i] * std[i] + mean[i]
    return img

def get_random_crop_positions_with_pos2d(img, pos2d, crop_size):
    max_pos = np.max(pos2d, axis=0)
    min_pos = np.min(pos2d, axis=0)

    if max_pos[0] - min_pos[0] > crop_size[0] or max_pos[1] - min_pos[1] > crop_size[1]:
        x_min = 0
        y_min = 0
        x_max = img.shape[1]
        y_max = img.shape[0]
    else:
        x_min_min = np.maximum(int(max_pos[0] - crop_size[0] + 50), 0)
        y_min_min = np.maximum(int(max_pos[1] - crop_size[1] + 50), 0)
        x_min_max = np.minimum(int(min_pos[0]) - 50, img.shape[1] - crop_size[0])
        y_min_max = np.minimum(int(min_pos[1]) - 50, img.shape[0] - crop_size[1])
        #print(f'max_pos: {max_pos}, min_pos: {min_pos}, x_min: {(x_min_min, x_min_max)}, y_min: {(y_min_min, y_min_max)}', flush=True)
        if y_min_min > y_min_max:
            #print(f'Wofule. y_min: {(y_min_min, y_min_max)}, max_pos: {max_pos}, min_pos: {min_pos}', flush=True)
            return 0, 0, img.shape[1], img.shape[0]
        x_min = np.random.randint(x_min_min, x_min_max) if x_min_min < x_min_max else x_min_min
        y_min = np.random.randint(y_min_min, y_min_max) if y_min_min < y_min_max else y_min_min
        x_max = x_min + crop_size[0]
        y_max = y_min + crop_size[1]

    return x_min, y_min, x_max, y_max

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2

def h36m_pos2d_preprocess(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D2_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,2)
        #np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts[:,Human36MMetadata.used_joint_mask,:])
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts)

def h36m_pos3d_preprocess(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D3_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,3)
        #np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts[:,Human36MMetadata.used_joint_mask,:])
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts)