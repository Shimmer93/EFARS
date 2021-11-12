import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_img_with_pos2d(img, pos2d, denormalize=None):
    img_array = img.numpy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = denormalize(img_array) if denormalize != None else img

    for pt in pos2d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img_array, pt, radius=0, color=(0, 255, 0), thickness=5)

    plt.imshow(img_array)
    plt.show()

def project_pos3d_to_pos2d(pos3d, R, t, cali):
    pos3d_homo = np.concatenate((pos3d, np.ones(pos3d.shape[0], 1)), axis=-1)
    P = cali @ np.hstack([R, t])
    pos2d_homo = pos3d_homo @ P.T
    pos2d = pos2d_homo[:,:2] / pos2d_homo[:,2:3]
    return pos2d

def show_img_with_pos3d(img, pos3d, R, t, cali, denormalize=None):
    pos2d = project_pos3d_to_pos2d(pos3d, R, t, cali)
    show_img_with_pos2d(img, pos2d, denormalize)

def show_img_with_hmap(img, hmap, denormalize=None):
    img_array = img.numpy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = denormalize(img_array) if denormalize != None else img

    hmap_array = hmap.numpy().sum(axis=0)

    ig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(img_array)
    axes[1].imshow(hmap_array)
    plt.show()