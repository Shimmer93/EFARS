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
    def __init__(self, img_fns, skeleton_2d_dir=None, skeleton_3d_dir=None, transforms=None, out_size=(256,256), downsample=8, sigma=3):
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
        self.sigma = sigma

    def __len__(self):
        return len(self.img_fns)

    def _get_img_info(self, index):
        img_fn = self.img_fns[index]

        fn_split = img_fn.split('_')
        subset = fn_split[0].split('/')[-1]
        action = fn_split[1] + ' ' + fn_split[2] if len(fn_split) == 4 else fn_split[1]
        frame = int(fn_split[-1].split('.')[0])

        return img_fn, subset, action, frame        
        
    def _prepare_img(self, img_fn):
        img = cv.imread(img_fn)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def _prepare_skeleton_2d(self, subset, action, frame):
        skeleton_2d_fn = os.path.join(self.skeleton_2d_dir, subset, 'MyPoseFeatures/D2_Positions', action + '.npy')
        skeleton_2ds = np.load(skeleton_2d_fn)
        if frame//5 < skeleton_2ds.shape[0]:
            skeleton_2d = skeleton_2ds[frame//5]
        else:
            skeleton_2d = np.copy(skeleton_2ds[-1])
        return skeleton_2d

    # Further processing is needed for 3D skeletons
    def _prepare_skeleton_3d(self, subset, action, frame):
        skeleton_3d_fn = os.path.join(self.skeleton_3d_dir, subset, 'MyPoseFeatures/D3_Positions', action.split('.')[0] + '.npy')
        skeleton_3ds = np.load(skeleton_3d_fn)
        if frame//5 < skeleton_3ds.shape[0]:
            skeleton_3d = skeleton_3ds[frame//5]
        else:
            skeleton_3d = np.copy(skeleton_3ds[-1].clone())
        return skeleton_3d

    def _generate_hmap2(self, skeleton_2d):
        hmap = np.zeros((skeleton_2d.shape[0], self.out_size[0], self.out_size[1]), dtype=float)
        for i in range(0, skeleton_2d.shape[0]):
            hmap[i, int(skeleton_2d[i,1])//self.downsample, int(skeleton_2d[i,0])//self.downsample] = 1.0
            hmap[i] = gaussian_filter(hmap[i], sigma=self.sigma)
        hmap[hmap > 1] = 1
        hmap[hmap < 0.001] = 0
        return hmap

    def _generate_hmap(self, skeleton_2d):
        hmap = np.zeros((self.out_size[0], self.out_size[1], skeleton_2d.shape[0]), dtype=float)
        for i in range(0, skeleton_2d.shape[0]):
            hmap[int(skeleton_2d[i,1]), int(skeleton_2d[i,0]), i] = 1.0
            hmap[:, :, i] = gaussian_filter(hmap[:, :, i], sigma=self.sigma)
        hmap[hmap > 1] = 1
        hmap[hmap < 0.001] = 0
        hmap = cv.resize(hmap, (self.out_size[0]//self.downsample, self.out_size[1]//self.downsample), cv.INTER_LINEAR)
        hmap = hmap.transpose(2, 0, 1)
        return hmap

class Human36M2DPoseDataset(Human36MBaseDataset):
    def __init__(self, img_fns, skeleton_2d_dir, transforms=None, crop_size=(512, 512), out_size=(256,256), downsample=8, sigma=3, mode='E'):
        super().__init__(img_fns=img_fns, skeleton_2d_dir=skeleton_2d_dir, transforms=transforms, out_size=out_size, downsample=downsample, sigma=sigma)
        self.transforms_params['crop_size'] = crop_size
        self.mode = mode

    def __getitem__(self, index):
        img_fn, subset, action, frame = self._get_img_info(index)
        img = self._prepare_img(img_fn)
        skeleton_2d = self._prepare_skeleton_2d(subset, action, frame)
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
        img_fn, subset, action, frame = self._get_img_info(index)
        img = self._prepare_img(img_fn)
        img = self.transforms(img, **self.transforms_params)
        skeleton_3d = self._prepare_skeleton_3d(subset, action, frame)
        return ToTensor()(img), torch.from_numpy(skeleton_3d)

class Human36M2DTo3DDataset(Human36MBaseDataset):
    def __init__(self, img_fns, skeleton_2d_dir, skeleton_3d_dir, transforms=None, out_size=(256,256), downsample=8):
        super().__init__(img_fns=img_fns, skeleton_2d_dir=skeleton_2d_dir, skeleton_3d_dir=skeleton_3d_dir, transforms=transforms, out_size=out_size, downsample=downsample)

    def __getitem__(self, index):
        _, subset, action, frame = self._get_img_info(index)
        skeleton_2d = self._prepare_skeleton_2d(subset, action, frame)
        skeleton_3d = self._prepare_skeleton_3d(subset, action, frame)
        return torch.from_numpy(skeleton_2d), torch.from_numpy(skeleton_3d)

class Human36M2DTemporalDataset(Human36MBaseDataset):
    def __init__(self, img_fns, skeleton_2d_dir, transforms=None, crop_size=(512, 512), out_size=(256,256), downsample=8, sigma=3, mode='E', length=5):
        super().__init__(img_fns=img_fns, skeleton_2d_dir=skeleton_2d_dir, transforms=transforms, out_size=out_size, downsample=downsample, sigma=sigma)
        self.transforms_params['crop_size'] = crop_size
        self.mode = mode
        self.length = length

    def __getitem__(self, index):
        img_fn, subset, action, frame = self._get_img_info(index)

        img_seq = []
        skeleton_2d_seq = []
        for i in range(self.length):
            img_fn_new = '_'.join(img_fn.split('_')[:-1]) + f'_{(frame+5*i):0>6d}.jpg'
            #print(f'The {i}-th image. Frame:{frame}, File name: {img_fn_new}, Origin name: {img_fn}')
            if img_fn_new in self.img_fns:
                img_new = self._prepare_img(img_fn_new)
            else:
                img_new = np.copy(img_seq[-1])
                    
            img_seq.append(img_new)

            skeleton_2d = self._prepare_skeleton_2d(subset, action, frame+5*i)
            skeleton_2d_seq.append(skeleton_2d)
    

        img_seq = np.stack(img_seq)
        skeleton_2d_seq = np.stack(skeleton_2d_seq)

        t, h, w, c = img_seq.shape
        img_seq = img_seq.transpose(1, 2, 0, 3).reshape(h, w, t*c)
        _, n, d = skeleton_2d_seq.shape
        skeleton_2d_seq = skeleton_2d_seq.reshape(t*n, d)
        img_seq, skeleton_2d_seq = self.transforms(img_seq, skeleton_2d_seq, **self.transforms_params)
        img_seq = img_seq.reshape(self.out_size[0], self.out_size[1], t, c).transpose(2, 0, 1, 3)
        skeleton_2d_seq = skeleton_2d_seq.reshape(t, n, d)

        if self.mode == 'estimation' or self.mode == 'E':
            hmap_seq = [self._generate_hmap(skeleton_2d_seq[i]) for i in range(t)]
            hmap_seq = np.stack(hmap_seq)
            return torch.from_numpy(img_seq.transpose(0, 3, 1, 2)), torch.from_numpy(skeleton_2d_seq), torch.from_numpy(hmap_seq)
        elif self.mode == 'classification' or self.mode == 'C':
            label = action.split('.')[0].split(' ')[0]
            label = Human36MMetadata.classes[label]
            return torch.from_numpy(img_seq.transpose(0, 3, 1, 2)), torch.tensor(label, dtype=torch.long)
        else:
            raise NotImplementedError

# TODO:
# class Human36MUnsupervised3DDataset(Human36MBaseDataset): # For pose embedding
# return 2 3D skeletons from the same video sequence with label "1" or from different video sequence with label "0"

if __name__ == '__main__':
    from glob import glob
    import sys
    sys.path.append('..')
    from utils.transform import do_pos2d_train_transforms
    img_fns = glob('/scratch/PI/cqf/datasets/h36m/img/*.jpg')
    train_dataset = Human36M2DTemporalDataset(img_fns, '/scratch/PI/cqf/datasets/h36m/pos2d', transforms=do_pos2d_train_transforms, mode='E')
    #img, hmap, _ = train_dataset[0]
    #print(img.shape)
    #print(hmap.shape)
    img, skeleton, hmap = train_dataset[907]
    print(img.shape)
    print(hmap.shape)