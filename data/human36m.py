import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from glob import glob
import os

class Human36M2dDataset(Dataset):
    def __init__(self, img_dir, pose_dir):
        super().__init__()
        self.img_fns = glob(os.path.join(img_dir, '*.jpg'))
        np.random.shuffle(self.img_fns)
        self.pose_dir = pose_dir
        #self.mask = np.array([])

    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        fn_split = img_fn.split('_')
        if len(fn_split) == 4:
            action = fn_split[1] + '_' + fn_split[2]
        else:
            action = fn_split[1]
        img = cv.imread(img_fn)
        pose_fn = os.path.join(self.pose_dir, fn_split[0].split('/')[-1], 'MyPoseFeatures/D2_Positions', action+'.npy')
        poses = np.load(pose_fn)
        pose = poses[int(fn_split[2].split('.')[0])//5]
        

        return img, pose






if __name__ == '__main__':
    train_dataset = Human36M2dDataset('/scratch/PI/cqf/datasets/h36m/img', '/scratch/PI/cqf/datasets/h36m/pos2d')
    img, pose = train_dataset[0]
    print(img.shape)
    print(pose.shape)