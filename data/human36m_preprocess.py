import cdflib
import numpy as np
from glob import glob
from tqdm import tqdm
import os
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv

USED_JOINT_MASK = np.array([1,1,1,1,0,0,1,1,
                            1,0,0,0,1,1,1,1,
                            0,1,1,1,0,0,0,0,
                            0,1,1,1,0,0,0,0],dtype=np.bool8)

def pos2d_preprocess(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D2_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,2)
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts[:,USED_JOINT_MASK,:])

def pos3d_preprocess(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D3_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,3)
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts[:,USED_JOINT_MASK,:])

'''
def generate_hmap(img_dir, pos2d_dir, output_dir):
    img_fns = glob(os.path.join(img_dir, '*.jpg'))
    for img_fn in img_fns:
        img = cv.imread(img_fn)
        w, h, c = img.shape
        hmap = np.zeros((w, h), dtype=float)


    pos2d_fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D2_Positions/*.npy'))
    for fn in tqdm(fns):
        pts = np.load(fn)
        for i, pt in enumerate(pts):
'''
    