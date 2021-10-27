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

def get_random_crop_positions_with_pos2d(img, pos2d, crop_size):
    max_pos = np.max(pos2d, axis=0)
    min_pos = np.min(pos2d, axis=0)

    if max_pos[0] - min_pos[0] > crop_size[0] or max_pos[1] - min_pos[1] > crop_size[1]:
        x_min = 0
        y_min = 0
        x_max = img.shape[0]
        y_max = img.shape[1]
    else:
        x_min_min = np.maximum(int(max_pos[0] - crop_size[0]) + 1, 0)
        y_min_min = np.maximum(int(max_pos[1] - crop_size[1]) + 1, 0)
        x_min_max = np.minimum(int(min_pos[0]), img.shape[0] - crop_size[0])
        y_min_max = np.minimum(int(min_pos[1]), img.shape[1] - crop_size[1])
        x_min = np.random.randint(x_min_min, x_min_max)
        y_min = np.random.randint(y_min_min, y_min_max)
        x_max = x_min + crop_size[0]
        y_max = y_min + crop_size[1]

    return x_min, y_min, x_max, y_max

def do_pos2d_train_transforms(img, pos2d, **kwargs):
    x_min, y_min, x_max, y_max = get_random_crop_positions_with_pos2d(img, pos2d, kwargs['crop_size'])
    transforms = A.Compose([
        A.Crop(x_min, y_min, x_max, y_max, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.Blur(blur_limit=3,p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Resize(height=kwargs['out_size'][0], width=kwargs['out_size'][1], p=1)
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pos2d)
    img = normalize(transformed['image'], kwargs['mean'], kwargs['std'])
    pos2d = np.array(transformed['keypoints'])
    return img, pos2d

def do_pos2d_val_transforms(img, pos2d, **kwargs):
    transforms = A.Compose([
        A.Resize(height=kwargs['out_size'][0], width=kwargs['out_size'][1], p=1),
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pos2d)
    img = normalize(transformed['image'], kwargs['mean'], kwargs['std'])
    pos2d = np.array(transformed['keypoints'])
    return img, pos2d