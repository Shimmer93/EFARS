import argparse
from glob import glob
from tqdm import tqdm
import cdflib
import os
import numpy as np

def h36m_pos2d_preprocess(data_dir):
    '''
        Convert pos2d data from .cdf to .npy and slice it to make it consistent with image data
    '''
    fns = glob(os.path.join(data_dir, 'S*/MyPoseFeatures/D2_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,2)
        np.save(os.path.join(data_dir, fn[len(data_dir):-4]), pts)

def h36m_pos3d_preprocess(data_dir):
    '''
        Convert pos3d data from .cdf to .npy and slice it to make it consistent with image data
    '''
    fns = glob(os.path.join(data_dir, 'S*/MyPoseFeatures/D3_Positions/*.cdf'))
    for fn in tqdm(fns):
        raw_data = cdflib.CDF(fn)
        pts = raw_data['Pose'][:,0::5,:].reshape(-1,32,3)
        np.save(os.path.join(data_dir, fn[len(data_dir):-4]), pts)

def h36m_preprocess(data_dir):
    subsets = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    pos2d_ids = ['1Z3Ve6CbxLzrJefuee-qY50m5D8ck5435', '1wZND_4_x8uCUFaE1ruX9fXm1E4jWyJUx', 
                 '1uNPfsSfDaqLZYH5AzagNzWk_ose8dez7', '1FhlFLxg-SmNWItk3L1hGpQ6bXkySRP8t',
                 '1SNKkcbaJGOKHHMa_emXYsX7M2HFC3piZ', '1cEDyCdKCICWBZmio1c92XW6ZMKrmZdoi',
                 '1pktxN4hu2Nle65TVNZYkeJ8vjTeGEPnS']
    pos3d_ids = ['1k2o4lUf3K_UIfzTpE5C60CoWUwbsRbgl', '1-BObud85jaxFB--E7GT1SZgfFvsvSmQB',
                 '1kc7njUyhQ_NqGgja5sy0YmpUNe5tbbBU', '1R1GWVVM82mCcIVKFH3GtF0h2p0UffQUN',
                 '1iWjTLKyHkV9wbUnBakb9A9sY5Qll7pJh', '1awPTBGKcsPzwJYJ9rXXAIsPJ-KAq2AIV',
                 '1Ek_NQGYdG4sv46aP0SHBElWQPFkc_4v-']

    for i in len(subsets):
        os.system(f'wget -P {data_dir}/imgs http://visiondata.cis.upenn.edu/volumetric/h36m/{subsets[i]}.tar')
        os.system(f'tar -xvf {data_dir}/imgs/{subsets[i]}.tar')
        os.system(f'rm {data_dir}/imgs/{subsets[i]}.tar')
        os.system(f'gdown {pos2d_ids[i]} -O {data_dir}/pos2d --folder')
        os.system(f'tar -zxvf {data_dir}/pos2d/Poses_D2_Positions_{subsets[i]}.tgz')
        os.system(f'rm {data_dir}/pos2d/Poses_D2_Positions_{subsets[i]}.tgz')
        os.system(f'gdown {pos3d_ids[i]} -O {data_dir}/pos3d --folder')
        os.system(f'tar -zxvf {data_dir}/pos2d/Poses_D3_Positions_{subsets[i]}.tgz')
        os.system(f'rm {data_dir}/pos3d/Poses_D3_Positions_{subsets[i]}.tgz')

    h36m_pos2d_preprocess(f'{data_dir}/pos2d')
    h36m_pos3d_preprocess(f'{data_dir}/pos3d')

def mm_fit_preprocess(data_dir):
    os.system(f'wget -P {data_dir} https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip')
    os.system(f'unzip -d {data_dir} mm-fit.zip')

def hmdb51_preprocess(data_dir):
    os.system(f'wget -P {data_dir}/videos http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar')
    os.system(f'wget -P {data_dir}/annots http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar')
    os.system(f'unrar e {data_dir}/videos {data_dir}/videos/hmdb51_org.rar')
    os.system(f'unrar e {data_dir}/annots {data_dir}/annots/test_train_splits.rar')
    os.system(f'rm {data_dir}/videos/hmdb51_org.rar')
    os.system(f'rm {data_dir}/annots/test_train_splits.rar')

    for files in os.listdir(f'{data_dir}/videos'):
        foldername = files.split('.')[0]
        os.system(f'mkdir -p {data_dir}/videos/{foldername}')
        os.system(f'unrar e {data_dir}/videos/{files} {data_dir}/videos/{foldername}')
        os.system(f'rm -rf {data_dir}/videos/{files}')

    to_remove = ['brush_hair','drink','pour','smile','sword','cartwheel','eat','smoke',
                 'wave','dive','fall_floor','hit','kiss','punch','talk','chew','hug',
                 'laugh','sit','stand','throw','clap','pick','shake_hands','turn']
    for foldername in to_remove:
        os.system(f'rm -rf {data_dir}/videos/{foldername}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h36m_dir', type=str, help='Path for Human3.6m dataset')
    parser.add_argument('--mm_fit_dir', type=str, help='Path for MM-Fit dataset')
    parser.add_argument('--hmdb_dir', type=str, help='Path for HMDB51 dataset')
    args = parser.parse_args()

    h36m_preprocess(args.h36m_dir)
    mm_fit_preprocess(args.mm_fit_dir)
    hmdb51_preprocess(args.hmdb_dir)
