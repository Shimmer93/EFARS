import sys
sys.path.insert(1, '/home/samuel/EFARS')
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from utils.conversion import project_pos3d_to_pos2d, hmap_to_pos2d

def prepare_img(img, denormalize=None):
    img_array = img.detach().numpy().copy()
    img_array = img_array.transpose(1, 2, 0)
    img_array = denormalize(img_array) if denormalize != None else img_array
    img_array = (img_array * 255).astype(np.uint8)
    return img_array

def show_img(img, denormalize=None):
    img_array = prepare_img(img, denormalize)
    plt.imshow(img_array)
    plt.show()

def show_img_with_pos2d(img, pos2d, denormalize=None, thickness=5):
    img_array = prepare_img(img, denormalize)

    for pt in pos2d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(img_array, pt, radius=0, color=(0, 255, 0), thickness=thickness)

    plt.imshow(img_array)
    plt.show()

def show_img_with_projected_pos3d(img, pos3d, project_matrix, denormalize=None):
    pos2d = project_pos3d_to_pos2d(pos3d, project_matrix)
    show_img_with_pos2d(img, pos2d, denormalize)

def show_pos2d_with_projected_pos3d(pos2d, pos3d, canvas_size, project_matrix, thickness=5):
    canvas_pos2d = np.zeros(canvas_size)
    canvas_pos3d = np.zeros(canvas_size)

    for pt in pos2d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(canvas_pos2d, pt, radius=0, color=(255, 255, 255), thickness=thickness)
    projected_pos3d = project_pos3d_to_pos2d(pos3d, project_matrix)
    for pt in projected_pos3d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(canvas_pos3d, pt, radius=0, color=(255, 255, 255), thickness=thickness)

    ig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(canvas_pos2d)
    axes[1].imshow(canvas_pos3d)
    plt.show()

def show_pos3d(pos3d, edges):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2])
    for edge in edges:
        data = np.stack((np.array(pos3d[edge[0]]), np.array(pos3d[edge[1]])))
        ax.plot3D(data[:,0], data[:,1], data[:,2], 'gray')
    plt.show()

def show_img_with_hmap(img, hmap, denormalize=None):
    img_array = prepare_img(img, denormalize)
    hmap_array = hmap.numpy().sum(axis=0)

    ig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(img_array)
    axes[1].imshow(hmap_array)
    plt.show()

def show_hmap_with_pos2d(hmap, img_size, pos2d=None, thickness=5):
    if pos2d == None:
        pos2d = hmap_to_pos2d(hmap, img_size)
    
    hmap_array = hmap.sum(axis=0)
    hmap_array = cv.resize(hmap_array, img_size)

    canvas_pos2d = np.zeros(img_size)
    for pt in pos2d:
        pt = (int(pt[0]), int(pt[1]))
        cv.circle(canvas_pos2d, pt, radius=0, color=(255, 255, 255), thickness=thickness)

    ig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(hmap_array)
    axes[1].imshow(canvas_pos2d)