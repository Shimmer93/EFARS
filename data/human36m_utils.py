import cdflib
import numpy as np
from glob import glob
from tqdm import tqdm
import os
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import json
from .human36m import Human36MMetadata

def pos2d_preprocess(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D2_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,2)
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts[:,Human36MMetadata.used_joint_mask,:])

def pos3d_preprocess(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    fns = glob(os.path.join(input_dir, 'S*/MyPoseFeatures/D3_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,3)
        np.save(os.path.join(output_dir, fn[len(input_dir):-4]), pts[:,Human36MMetadata.used_joint_mask,:])

def get_project_matrix(cps, subset, id):
    Rt = cps['extrinsics'][subset][str(id)]
    R = Rt['R']
    t = Rt['t']
    cali = cps['intrinsics'][str(id)]['calibration_matrix']
    P = cali @ np.hstack([R, t])

    return P