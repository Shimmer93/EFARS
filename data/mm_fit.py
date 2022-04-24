import numpy as np
import torch
from torch.utils.data import Dataset
import csv

class MMFitMetaData:
    num_classes = 11
    classes = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5,
               'dumbbell_rows': 6, 'jumping_jacks': 7, 'dumbbell_shoulder_press': 8,
               'lateral_shoulder_raises': 9, 'non_activity': 10}
    num_joints = 17
    joint_labels = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 
                         'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 
                         'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    skeleton_edges = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 4), (6, 5), (7, 0), (8, 7), 
                      (9, 8), (10, 9), (11, 8), (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)]


# sample_poses: C * T * V * 1
# labels: int
# reps: int
class MMFit(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """
    def __init__(self, pose_3d_path, label_path, skeleton_window_length, stride=5):
        """
        Initialize MMFit Dataset object.
        :param pose_3d_path: File path to 3D pose file for a workout.
        :param label_path: File path to MM-Fit CSV label file for a workout.
        :param skeleton_window_length: Skeleton window length in number of samples.
        :param stride: Stride of frames
        """
        
        self.skeleton_window_length = skeleton_window_length
        self.poses = self._load_modality(pose_3d_path)
        self.labels = self._load_labels(label_path)
        self.stride = stride

    def __len__(self):
        return self.poses.shape[1] - self.skeleton_window_length - 30
        
    def __getitem__(self, i):
        frame = self.poses[0, i, 0]
        sample_poses = torch.as_tensor(self.poses[:, i:i+self.skeleton_window_length*self.stride:self.stride, 1:], dtype=torch.float)
        sample_poses -= sample_poses[:,:,:1]
        sample_poses /= 1000
        sample_poses = sample_poses.unsqueeze(-1)

        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame > (row[0] - self.skeleton_window_length/2)) and (frame < (row[1] - self.skeleton_window_length/2)):
                label = row[3]
                reps = row[2]
                break
        
        return sample_poses, MMFitMetaData.classes[label], reps

    def _load_modality(self, filepath):
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

    def _load_labels(self, filepath):
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