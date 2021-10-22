import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from glob import glob
import os

#CLASSES = []

#CHANNEL_MEAN = []


class Human36MBaseDataset(Dataset):
    def __init__(self, img_dir, subsets, skeleton_2d_dir=None, skeleton_3d_dir=None, transforms=None, img_size=(256,256)):
        super().__init__()
        self.img_dir = img_dir
        self.img_fns = []
        for subset in subsets:
            self.img_fns += glob(os.path.join(img_dir, f'{subset}*.jpg'))
        np.random.shuffle(self.img_fns)
        self.skeleton_2d_dir = skeleton_2d_dir
        self.skeleton_3d_dir = skeleton_3d_dir
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.img_fns)

    def _get_img_info(self, index):
        img_fn = self.img_fns[index]
        img = cv.imread(img_fn)

        fn_split = img_fn.split('_')
        subset = fn_split[0].split('/')[-1]
        action = fn_split[1] + ' ' + fn_split[2] if len(fn_split) == 4 else fn_split[1]
        frame = int(fn_split[2].split('.')[0])

        return img, subset, action, frame
        
    def _prepare_img(self, img):
        if self.img_size != None:
            img = cv.resize(img, self.img_size, interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        img = img.float() / 255.0
        if self.transforms != None:
            img = self.transforms(img)
        return img

    def _prepare_skeleton_2d(self, subset, action, frame, img_orig_size):
        skeleton_2d_fn = os.path.join(self.skeleton_2d_dir, subset, 'MyPoseFeatures/D2_Positions', action + '.npy')
        skeleton_2ds = np.load(skeleton_2d_fn)
        skeleton_2d = skeleton_2ds[frame//5]
        #skeleton_2d = skeleton_2d[USED_JOINT_MASK,:]
        skeleton_2d[:, 0] *= self.img_size[0] / img_orig_size[0]
        skeleton_2d[:, 1] *= self.img_size[1] / img_orig_size[1]
        
        return torch.from_numpy(skeleton_2d)

    # Further processing is needed for 3D skeletons
    def _prepare_skeleton_3d(self, subset, action, frame, img_orig_size):
        skeleton_3d_fn = os.path.join(self.skeleton_3d_dir, subset, 'MyPoseFeatures/D3_Positions', action.split('.')[0] + '.npy')
        skeleton_3ds = np.load(skeleton_3d_fn)
        skeleton_3d = skeleton_3ds[frame//5]
        #skeleton_3d = skeleton_3d[USED_JOINT_MASK,:]
        skeleton_3d[:, 0] *= self.img_size[0] / img_orig_size[0]
        skeleton_3d[:, 1] *= self.img_size[1] / img_orig_size[1]
        return torch.from_numpy(skeleton_3d)

class Human36M2DPoseDataset(Human36MBaseDataset):
    def __init__(self, img_dir, subsets, skeleton_2d_dir, transforms=None, img_size=(256,256)):
        super().__init__(img_dir=img_dir, subsets=subsets, skeleton_2d_dir=skeleton_2d_dir, transforms=transforms, img_size=img_size)

    def __getitem__(self, index):
        img, subset, action, frame = self._get_img_info(index)
        w, h, _ = img.shape
        img_orig_size = (w, h)
        img = self._prepare_img(img)
        skeleton_2d = self._prepare_skeleton_2d(subset, action, frame, img_orig_size)
        return img, skeleton_2d

class Human36M3DPoseDataset(Human36MBaseDataset):
    def __init__(self, img_dir, subsets, skeleton_3d_dir, transforms=None, img_size=(256,256)):
        super().__init__(img_dir=img_dir, subsets=subsets, skeleton_3d_dir=skeleton_3d_dir, transforms=transforms, img_size=img_size)

    def __getitem__(self, index):
        img, subset, action, frame = self._get_img_info(index)
        w, h, _ = img.shape
        img_orig_size = (w, h)
        img = self._prepare_img(img)
        skeleton_3d = self._prepare_skeleton_3d(subset, action, frame, img_orig_size)
        return img, skeleton_3d

class Human36M2DTo3DDataset(Human36MBaseDataset):
    def __init__(self, img_dir, subsets, skeleton_2d_dir, skeleton_3d_dir, transforms=None, img_size=(256,256)):
        super().__init__(img_dir=img_dir, subsets=subsets, skeleton_2d_dir=skeleton_2d_dir, skeleton_3d_dir=skeleton_3d_dir, transforms=transforms, img_size=img_size)

    def __getitem__(self, index):
        img, subset, action, frame = self._get_img_info(index)
        w, h, _ = img.shape
        img_orig_size = (w, h)
        skeleton_2d = self._prepare_skeleton_2d(subset, action, frame, img_orig_size)
        skeleton_3d = self._prepare_skeleton_3d(subset, action, frame, img_orig_size)
        return skeleton_2d, skeleton_3d


# TODO:
# class Human36MUnsupervised3DDataset(Human36MBaseDataset): # For pose embedding
# return 2 3D skeletons from the same video sequence with label "1" or from different video sequence with label "0"

if __name__ == '__main__':
    train_dataset = Human36M2DPoseDataset('/scratch/PI/cqf/datasets/h36m/img', ['S1', 'S5'], '/scratch/PI/cqf/datasets/h36m/pos2d')
    img, skeleton = train_dataset[0]
    print(img.shape)
    print(skeleton.shape)
    print(skeleton)