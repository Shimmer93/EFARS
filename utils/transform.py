from numpy.ma.core import putmask
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch

def normalize(img, mean, std):
    for i in range(3):
        img[i,:,:] = (img[i,:,:] - mean[i]) / std[i]
    return img

def denormalize(img, mean, std):
    for i in range(3):
        img[i,:,:] = img[i,:,:] * std[i] + mean[i]
    return img


def do_pos2d_train_transforms(img, pts, crop_size, img_size, mean, std):
    max_pos = np.max(pts, axis=0)
    min_pos = np.min(pts, axis=1)
    min_w = np.maximum(int(max_pos[0] - crop_size[0]), 0)
    min_h = np.maximum(int(max_pos[1] - crop_size[1]), 0)
    max_w = np.minimum(int(min_pos[0]) + 1, img.shape[0] - crop_size[0]) + 1
    max_h = np.minimum(int(min_pos[1]) + 1, img.shape[1] - crop_size[1]) + 1
    rand_w = np.random.randint(min_w, max_w)
    rand_h = np.random.randint(min_h, max_h)

    transforms = A.Compose([
        A.Crop(rand_w, rand_h, rand_w+crop_size[0], rand_h+crop_size[1], p=1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.Blur(blur_limit=3,p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Resize(height=img_size[0], width=img_size[1], p=1),
        ToTensorV2(p=1)
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pts)
    img = normalize(transformed['image'], mean, std)
    pts = torch.tensor(transformed['keypoints'])
    return img, pts

def do_img_only_transforms(img, mean, std):
    transforms = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.Blur(blur_limit=3,p=0.2),
        ToTensorV2(p=1)
    ])
    img = transforms(img)['image']
    img = normalize(img, mean, std)
    return img