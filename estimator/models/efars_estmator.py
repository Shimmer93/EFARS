import torch
import torch.nn as nn
import sys
sys.path.append('../..')
from utils.misc import get_max_preds

class PoseEstmator(nn.Module):
    def __init__(self, pos2dModel, pos3dModel, numJoint):
        if pos2dModel == 'openpose':
            from .openpose import OpenPose
            self.frontend = OpenPose()
            self.frontend_metadata = {'mode': 'heat_map', 'downsample': 8}
        elif pos2dModel == 'unipose':
            from .unipose import UniPose
            self.frontend = UniPose(dataset='human3.6m', num_classes=numJoint)
            self.frontend_metadata = {'mode': 'heat_map', 'downsample': 8}
        else:
            raise NotImplementedError
        self.frontend.eval()

        if pos3dModel == 'pose2mesh':
            from .pose2mesh import PoseNet
            self.backend = PoseNet(num_joint=numJoint)
            self.backend_metadata = {'mode': 'positions'}
        else:
            raise NotImplementedError
        self.backend.eval()

    def forward(self, x):
        h = self.frontend(x)
        if self.frontend_metadata['mode'] == self.backend_metadata['mode']:
            y = self.backend(h)
            return y
        elif self.frontend_metadata['mode'] == 'heat_map' and self.backend_metadata['mode'] == 'positions':
            h, _ = get_max_preds(h)
            y = self.backend(h)
            return y
        else:
            raise NotImplementedError
            