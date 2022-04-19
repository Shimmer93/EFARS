import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import csv

def load_modality(filepath):
    """
    Loads modality from filepath and returns numpy array, or None if no file is found.
    :param filepath: File path to MM-Fit modality.
    :return: MM-Fit modality (numpy array).
    """
    try:
        mod = np.load(filepath)
    except FileNotFoundError as e:
        mod = None
        print('{}. Returning None'.format(e))
    return mod


def load_labels(filepath):
    """
    Loads and reads CSV MM-Fit CSV label file.
    :param filepath: File path to a MM-Fit CSV label file.
    :return: List of lists containing label data, (Start Frame, End Frame, Repetition Count, Activity) for each
    exercise set.
    """
    labels = []
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            labels.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
    return labels

class MMFit(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, pose_3d_path, label_path, skeleton_window_length, stride=5):
        """
        Initialize MMFit Dataset object.
        :param modality_filepaths: Modality - file path mapping (dict) for a workout.
        :param label_path: File path to MM-Fit CSV label file for a workout.
        :param window_length: Window length in seconds.
        :param skeleton_window_length: Skeleton window length in number of samples.
        :param sensor_window_length: Sensor window length in number of samples.
        :param skeleton_transform: Transformation functions to apply to skeleton data.
        :param sensor_transform: Transformation functions to apply to sensor data.
        """
        
        self.skeleton_window_length = skeleton_window_length
    
        self.poses = load_modality(pose_3d_path)

        self.ACTIONS = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5,
                        'dumbbell_rows': 6, 'jumping_jacks': 7, 'dumbbell_shoulder_press': 8,
                        'lateral_shoulder_raises': 9, 'non_activity': 10}
        self.labels = load_labels(label_path)
        self.stride = stride

    def __len__(self):
        return self.poses.shape[1] - self.skeleton_window_length - 30
        
    def __getitem__(self, i):
        frame = self.poses[0, i, 0]
        sample_poses = torch.as_tensor(self.poses[:, i:i+self.skeleton_window_length*self.stride:self.stride, 1:], dtype=torch.float)
        sample_poses = sample_poses.permute(1, 2, 0)
        sample_poses -= sample_poses[:,:1,:]
        sample_poses /= 1000
        sample_poses = sample_poses.permute(2, 0, 1).unsqueeze(-1) # C * T * V * 1

        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame > (row[0] - self.skeleton_window_length/2)) and (frame < (row[1] - self.skeleton_window_length/2)):
                label = row[3]
                reps = row[2]
                break
        
        return sample_poses, self.ACTIONS[label], reps


class SequentialStridedSampler(Sampler):
    """
    PyTorch Sampler Class to sample elements sequentially using a specified stride, always in the same order.
    Arguments:
        data_source (Dataset):
        stride (int):
    """

    def __init__(self, data_source, stride):
        """
        Initialize SequentialStridedSampler object.
        :param data_source: Dataset to sample from.
        :param stride: Stride to slide window in seconds.
        """
        self.data_source = data_source
        self.stride = stride
    
    def __len__(self):
        return len(range(0, len(self.data_source), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.stride))

if __name__ == '__main__':
    ds = MMFit('/home/samuel/mm-fit/w00/w00_pose_3d.npy', '/home/samuel/mm-fit/w00/w00_labels.csv', 32)
    m, l, r = ds[0]
    print(m)
    #print(l)
    #print(r)