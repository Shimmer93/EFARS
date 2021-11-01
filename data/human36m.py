import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import cv2 as cv
from glob import glob
import os
from scipy.ndimage import gaussian_filter
import json

class Human36MMetadata:
    num_classes = 14
    classes = {'Directions': 0, 'Discussion': 1, 'Eating': 2, 'Greeting': 3, 'TakingPhoto': 4, 'Photo':4, 
               'Posing': 5, 'Purchases': 6, 'Smoking': 7, 'Waiting': 8, 'Walking': 9, 'Sitting': 10, 
               'SittingDown': 10, 'Phoning': 11, 'WalkingDog': 12, 'WalkDog': 12, 'WalkTogether': 13}
    mean = np.array([0.44245931, 0.2762126, 0.2607548])
    std = np.array([0.25389833, 0.26563732, 0.24224165])
    num_joints = 17
    used_joint_mask = np.array([1,1,1,1,0,0,1,1,
                                1,0,0,0,1,1,1,1,
                                0,1,1,1,0,0,0,0,
                                0,1,1,1,0,0,0,0],dtype=np.bool8)
    used_joint_labels = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 
                         'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 
                         'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    skeleton_edges = [(10, 9), (9, 8), (8, 11), (11, 12), (12, 13), (8, 14), 
                      (14, 15), (15, 16), (8, 7), (7, 0), (0, 1), (1, 2), (2, 3), 
                      (0, 4), (4, 5), (5, 6)]
    #camera_parameters = json.load(open('human36m_camera_parameters.json', 'r'))


class Human36MBaseDataset(Dataset):
    def __init__(self, img_fns, skeleton_2d_dir=None, skeleton_3d_dir=None, transforms=None, out_size=(256,256), downsample=8):
        super().__init__()
        self.img_fns = img_fns
        self.skeleton_2d_dir = skeleton_2d_dir
        self.skeleton_3d_dir = skeleton_3d_dir
        self.transforms = transforms
        self.transforms_params = {
            'out_size': out_size,
            'mean': Human36MMetadata.mean,
            'std': Human36MMetadata.std
        }
        self.out_size = out_size
        self.downsample = downsample

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
        #if self.out_size != None:
        #    img = cv.resize(img, self.out_size, interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #img = torch.from_numpy(img.transpose((2, 0, 1)))
        img = img.astype(np.float32) / 255.0
        return img

    def _prepare_skeleton_2d(self, subset, action, frame, img_orig_size):
        skeleton_2d_fn = os.path.join(self.skeleton_2d_dir, subset, 'MyPoseFeatures/D2_Positions', action + '.npy')
        skeleton_2ds = np.load(skeleton_2d_fn)
        skeleton_2d = skeleton_2ds[frame//5]
        #skeleton_2d = skeleton_2d[USED_JOINT_MASK,:]
        skeleton_2d[:, 0] *= self.out_size[0] / img_orig_size[0]
        skeleton_2d[:, 1] *= self.out_size[1] / img_orig_size[1]
        return skeleton_2d

    # Further processing is needed for 3D skeletons
    def _prepare_skeleton_3d(self, subset, action, frame, img_orig_size):
        skeleton_3d_fn = os.path.join(self.skeleton_3d_dir, subset, 'MyPoseFeatures/D3_Positions', action.split('.')[0] + '.npy')
        skeleton_3ds = np.load(skeleton_3d_fn)
        skeleton_3d = skeleton_3ds[frame//5]
        #skeleton_3d = skeleton_3d[USED_JOINT_MASK,:]
        skeleton_3d[:, 0] *= self.out_size[0] / img_orig_size[0]
        skeleton_3d[:, 1] *= self.out_size[1] / img_orig_size[1]
        return skeleton_3d

    def _generate_hmap(self, skeleton_2d):
        hmap = np.zeros((skeleton_2d.shape[0]+1, self.out_size[0]//self.downsample, self.out_size[1]//self.downsample), dtype=float)
        for i in range(0, skeleton_2d.shape[0]):
            hmap[i+1, int(skeleton_2d[i,0])//self.downsample, int(skeleton_2d[i,1])//self.downsample] = 1.0
            hmap[i+1] = gaussian_filter(hmap[i+1], sigma=3)
        hmap[hmap > 1] = 1
        hmap[hmap < 0.0099] = 0
        hmap[0] = 1.0 - np.max(hmap[1:, :, :], axis=0)
        return hmap

class Human36M2DPoseDataset(Human36MBaseDataset):
    def __init__(self, img_fns, skeleton_2d_dir, transforms=None, crop_size=(512, 512), out_size=(256,256), downsample=8, mode='E'):
        super().__init__(img_fns=img_fns, skeleton_2d_dir=skeleton_2d_dir, transforms=transforms, out_size=out_size, downsample=downsample)
        self.transforms_params['crop_size'] = crop_size
        self.mode = mode

    def __getitem__(self, index):
        img, subset, action, frame = self._get_img_info(index)
        w, h, _ = img.shape
        img_orig_size = (w, h)
        img = self._prepare_img(img)
        skeleton_2d = self._prepare_skeleton_2d(subset, action, frame, img_orig_size)
        img, skeleton_2d = self.transforms(img, skeleton_2d, **self.transforms_params)
        if self.mode == 'estimation' or self.mode == 'E':
            hmap = self._generate_hmap(skeleton_2d)
            return ToTensor()(img), torch.from_numpy(hmap), torch.from_numpy(skeleton_2d)
        elif self.mode == 'classification' or self.mode == 'C':
            label = action.split('.')[0].split(' ')[0]
            label = Human36MMetadata.classes[label]
            return ToTensor()(img), torch.from_numpy(skeleton_2d), torch.tensor(label, dtype=torch.long)
        else:
            raise NotImplementedError

class Human36M3DPoseDataset(Human36MBaseDataset):
    def __init__(self, img_fns, skeleton_3d_dir, transforms=None, out_size=(256,256), downsample=8):
        super().__init__(img_fns=img_fns, skeleton_3d_dir=skeleton_3d_dir, transforms=transforms, out_size=out_size, downsample=downsample)

    def __getitem__(self, index):
        img, subset, action, frame = self._get_img_info(index)
        w, h, _ = img.shape
        img_orig_size = (w, h)
        img = self._prepare_img(img)
        img = self.transforms(img, **self.transforms_params)
        skeleton_3d = self._prepare_skeleton_3d(subset, action, frame, img_orig_size)
        return ToTensor()(img), torch.from_numpy(skeleton_3d)

class Human36M2DTo3DDataset(Human36MBaseDataset):
    def __init__(self, img_fns, skeleton_2d_dir, skeleton_3d_dir, transforms=None, out_size=(256,256), downsample=8):
        super().__init__(img_fns=img_fns, skeleton_2d_dir=skeleton_2d_dir, skeleton_3d_dir=skeleton_3d_dir, transforms=transforms, out_size=out_size, downsample=downsample)

    def __getitem__(self, index):
        img, subset, action, frame = self._get_img_info(index)
        w, h, _ = img.shape
        img_orig_size = (w, h)
        skeleton_2d = self._prepare_skeleton_2d(subset, action, frame, img_orig_size)
        skeleton_3d = self._prepare_skeleton_3d(subset, action, frame, img_orig_size)
        return torch.from_numpy(skeleton_2d), torch.from_numpy(skeleton_3d)

# TODO:
# class Human36MUnsupervised3DDataset(Human36MBaseDataset): # For pose embedding
# return 2 3D skeletons from the same video sequence with label "1" or from different video sequence with label "0"

if __name__ == '__main__':
    from glob import glob
    import sys
    sys.path.append('..')
    from utils.transform import do_pos2d_train_transforms
    img_fns = glob('/scratch/PI/cqf/datasets/h36m/img/*.jpg')
    train_dataset = Human36M2DPoseDataset(img_fns, '/scratch/PI/cqf/datasets/h36m/pos2d', transforms=do_pos2d_train_transforms, mode='C')
    #img, hmap, _ = train_dataset[0]
    #print(img.shape)
    #print(hmap.shape)
    img, label = train_dataset[0]
    print(img.shape)
    print(label)