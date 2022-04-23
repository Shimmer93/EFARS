import numpy as np
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter

def get_project_matrix(R, t, cali):
    return cali @ np.hstack([R, t])

def project_pos3d_to_pos2d(pos3d, project_matrix):
    pos3d_homo = np.concatenate((pos3d, np.ones((pos3d.shape[0], 1))), axis=-1)
    pos2d_homo = pos3d_homo @ project_matrix.T
    pos2d = pos2d_homo[:,:2] / pos2d_homo[:,2:3]
    return pos2d

def pos2d_to_hmap(pos2d, img_size, downsample, sigma):
    hmap = np.zeros((img_size[0], img_size[1], pos2d.shape[0]), dtype=float)
    for i in range(0, pos2d.shape[0]):
        hmap[int(pos2d[i,1]), int(pos2d[i,0]), i] = 1.0
        hmap[:, :, i] = gaussian_filter(hmap[:, :, i], sigma=sigma)
    hmap[hmap > 1] = 1
    hmap[hmap < 0.001] = 0
    hmap = cv.resize(hmap, (img_size[0]//downsample, img_size[1]//downsample), cv.INTER_LINEAR)
    hmap = hmap.transpose(2, 0, 1)
    return hmap

def hmap_to_pos2d(hmap, img_size):
    pos2d = []
    for i in range(len(hmap)):
        resized_hmap = cv.resize(hmap[i], img_size, interpolation=cv.INTER_CUBIC)
        pos = np.unravel_index(np.argmax(resized_hmap, axis=None), resized_hmap.shape)
        x = int(pos[1])
        y = int(pos[0])
        pos2d.append([x, y])
    pos2d = np.array(pos2d)
    return pos2d