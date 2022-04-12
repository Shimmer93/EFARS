import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch

#def normalize(img, mean, std):
#    img = (img - std[None,None,...]) / mean[None,None,...]
#    return img

#def denormalize(img, mean, std):
#    img = img * std[None,None,...] + mean[None,None,...]
#    return img

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

def do_pos2d_train_transforms(img, pos2d, **kwargs):
    #x_min, y_min, x_max, y_max = get_random_crop_positions_with_pos2d(img, pos2d, kwargs['crop_size'])
    transforms = A.Compose([
        #A.Crop(x_min, y_min, x_max, y_max, p=0.5),
        #A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        #A.Blur(blur_limit=3,p=0.2),
        #A.HorizontalFlip(p=0.5),
        A.Resize(height=kwargs['out_size'][0], width=kwargs['out_size'][1], p=1)
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pos2d)
    img = transformed['image']
    if img.shape[-1] == 3:
        img = normalize(img, kwargs['mean'], kwargs['std'])
    elif img.shape[-1] % 3 == 0:
        for i in range(img.shape[-1] // 3):
            img[:, :, 3*i:3*i+3] = normalize(img[:, :, 3*i:3*i+3], kwargs['mean'], kwargs['std'])
    else:
        raise ValueError
    pos2d = np.array(transformed['keypoints'])
    return img, pos2d

def do_pos2d_val_transforms(img, pos2d, **kwargs):
    transforms = A.Compose([
        A.Resize(height=kwargs['out_size'][0], width=kwargs['out_size'][1], p=1),
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transforms(image=img, keypoints=pos2d)
    img = transformed['image']
    if img.shape[-1] == 3:
        img = normalize(img, kwargs['mean'], kwargs['std'])
    elif img.shape[-1] % 3 == 0:
        for i in range(img.shape[-1] // 3):
            img[:, :, 3*i:3*i+3] = normalize(img[:, :, 3*i:3*i+3], kwargs['mean'], kwargs['std'])
    else:
        raise ValueError
    pos2d = np.array(transformed['keypoints'])
    return img, pos2d

def do_2d_to_3d_transforms(pos2d, pos3d, **kwargs):
    pos2d = normalize(pos2d, kwargs['pos2d_mean'], kwargs['pos2d_std'])
    pos3d = pos3d / 1000 #normalize(pos3d, kwargs['pos3d_mean'], kwargs['pos3d_std'])
    return pos2d, pos3d