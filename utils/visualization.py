import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt

def show_img_with_2d_keypoints(img, pts, denormalize=None):
    img_tensor = denormalize(img) if denormalize != None else img
    img_array = (img_tensor.numpy() * 255.0).astype(np.uint8)
    img_array = img_array.transpose(1, 2, 0)

    for pt in pts[0]:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img_array, pt, radius=0, color=(0, 255, 0), thickness=5)

    plt.imshow(img_array)
    plt.show()
    