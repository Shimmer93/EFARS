# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from model import model_zoo
from base.base_model import base_model
from utils import util,h36m_utils


class fk_model(base_model):
    def __init__(self, config):
        super(fk_model, self).__init__()
        self.config = config
        assert len(config.arch.kernel_size) == len(config.arch.stride) == len(config.arch.dilation)
        self.parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.rotation_type = config.arch.rotation_type
        self.rotation_number = util.ROTATION_NUMBERS.get(config.arch.rotation_type)
        self.EQ = model_zoo.EQ_net(34, 48)
        self.fk_layer = model_zoo.fk_layer(config.arch.rotation_type)
        self.optimizer_Q = torch.optim.Adam(list(self.EQ.parameters()), lr=config.trainer.lr, amsgrad=True)
        print('Building the network')

    def forward_Q(self, _input):
        return self.EQ(_input)[:, :, :12*self.rotation_number]

    def forward_fk(self, _input, bones, norm_parameters, cam_para):
        # print(_input.shape)
        con_skeleton = condition_skeleton(_input.shape[0], norm_parameters[2])
        # print(con_skeleton.shape)
        src_skeleton = source_skeleton(bones, norm_parameters[2], norm_parameters[3])
        # print(src_skeleton.shape)
        output_EQ = self.EQ(_input)
        # print(output_EQ.shape)
        fake_rotations = output_EQ[:, :, :12*self.rotation_number]
        fake_rotations_full = torch.zeros((fake_rotations.shape[0], fake_rotations.shape[1], 17*self.rotation_number), requires_grad=True).cuda()
        fake_rotations_full[:, :, np.arange(17)*self.rotation_number] = 1 if self.rotation_type == 'q' else 0# Set all to identity quaternion
        complate_indices = np.sort(np.hstack([np.array([0,1,2,4,5,7,8,9,11,12,14,15])*self.rotation_number + i for i in range(self.rotation_number)]))
        fake_rotations_full[:,:,complate_indices] = fake_rotations
        con_pose_3d = self.fk_layer.forward(self.parents, con_skeleton.repeat(_input.shape[1], 1, 1), fake_rotations_full.contiguous().view(-1, 17, self.rotation_number)).view(_input.shape[0], _input.shape[1], -1)
        src_pose_3d = self.fk_layer.forward(self.parents, src_skeleton.repeat(_input.shape[1], 1, 1), fake_rotations_full.contiguous().view(-1, 17, self.rotation_number)).view(_input.shape[0], _input.shape[1], -1)
        # print(con_pose_3d.shape)
        # print(src_pose_3d.shape)
        con_pose_2d_project = torch.zeros(_input.shape)
        for b in range(_input.shape[0]):
          for n in range(_input.shape[1]):
            R = cam_para[b, n, :9].view((3, 3)).detach().cpu().numpy()
            T = cam_para[b, n, 9:12].view((3, 1)).detach().cpu().numpy()
            f = cam_para[b, n, 12:14].view((2, 1)).detach().cpu().numpy()
            c = cam_para[b, n, 14:16].view((2, 1)).detach().cpu().numpy()
            k = cam_para[b, n, 16:19].view((3, 1)).detach().cpu().numpy()
            p = cam_para[b, n, 19:21].view((2, 1)).detach().cpu().numpy()
            set_3d = con_pose_3d[b, n, :].view((-1, 3)).detach().cpu().numpy()
            set_2d_project = abs(np.nan_to_num(h36m_utils.project_2d(set_3d.reshape((-1, 3)), R, T, f, c, k, p, from_world=False)[0].flatten()))
            set_2d_project = _input[b, n, :].detach().cpu().numpy() + set_2d_project / np.max(set_2d_project)
            con_pose_2d_project[b, n, :] = torch.from_numpy(set_2d_project)
        output_EQ = self.EQ(con_pose_2d_project.float().cuda())
        fake_rotations_cycle = output_EQ[:, :, :12*self.rotation_number]
        fake_rotations_full = torch.zeros((fake_rotations_cycle.shape[0], fake_rotations_cycle.shape[1], 17*self.rotation_number), requires_grad=True).cuda()
        fake_rotations_full[:, :, np.arange(17)*self.rotation_number] = 1 if self.rotation_type == 'q' else 0# Set all to identity quaternion
        complate_indices = np.sort(np.hstack([np.array([0,1,2,4,5,7,8,9,11,12,14,15])*self.rotation_number + i for i in range(self.rotation_number)]))
        fake_rotations_full[:,:,complate_indices] = fake_rotations_cycle
        src_pose_3d_cycle = self.fk_layer.forward(self.parents, src_skeleton.repeat(_input.shape[1], 1, 1), fake_rotations_full.contiguous().view(-1, 17, self.rotation_number)).view(_input.shape[0], _input.shape[1], -1)
        return fake_rotations, fake_rotations_cycle, src_pose_3d, src_pose_3d_cycle

    def lr_decaying(self, decay_rate):
        optimizer_set = [self.optimizer_length, self.optimizer_rotation]
        for optimizer in optimizer_set:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay_rate


def distance(position1, position2):
    vector = torch.abs(position1 - position2)
    return torch.mean(torch.sqrt(torch.sum(torch.pow(vector, 2), dim=-1)), dim=-1)


def get_velocity(motions, joint_index):
    joint_motion = motions[..., [joint_index*3, joint_index*3 + 1, joint_index*3 + 2]]
    velocity = torch.sqrt(torch.sum((joint_motion[:, 2:] - joint_motion[:, :-2])**2, dim=-1))
    return velocity

def condition_skeleton(in_shape, bone_mean):
    unnorm_bones = bone_mean.repeat(in_shape, 1, 1)
    skel_in = torch.zeros(in_shape, 17, 3).cuda()
    skel_in[:, 1, 0] = -unnorm_bones[:, 0, 0]
    skel_in[:, 4, 0] = unnorm_bones[:, 0, 0]
    skel_in[:, 2, 1] = -unnorm_bones[:, 0, 1]
    skel_in[:, 5, 1] = -unnorm_bones[:, 0, 1]
    skel_in[:, 3, 1] = -unnorm_bones[:, 0, 2]
    skel_in[:, 6, 1] = -unnorm_bones[:, 0, 2]
    skel_in[:, 7, 1] = unnorm_bones[:, 0, 3]
    skel_in[:, 8, 1] = unnorm_bones[:, 0, 4]
    skel_in[:, 9, 1] = unnorm_bones[:, 0, 5]
    skel_in[:, 10, 1] = unnorm_bones[:, 0, 6]
    skel_in[:, 11, 0] = unnorm_bones[:, 0, 7]
    skel_in[:, 12, 0] = unnorm_bones[:, 0, 8]
    skel_in[:, 13, 0] = unnorm_bones[:, 0, 9]
    skel_in[:, 14, 0] = -unnorm_bones[:, 0, 7]
    skel_in[:, 15, 0] = -unnorm_bones[:, 0, 8]
    skel_in[:, 16, 0] = -unnorm_bones[:, 0, 9]
    return skel_in


def source_skeleton(bones, bone_mean, bone_std):
    bones = torch.mean(bones, dim=1)
    unnorm_bones = bones * bone_std.unsqueeze(0) + bone_mean.repeat(bones.shape[0], 1, 1)
    skel_in = torch.zeros(bones.shape[0], 17, 3).cuda()
    skel_in[:, 1, 0] = -unnorm_bones[:, 0, 0]
    skel_in[:, 4, 0] = unnorm_bones[:, 0, 0]
    skel_in[:, 2, 1] = -unnorm_bones[:, 0, 1]
    skel_in[:, 5, 1] = -unnorm_bones[:, 0, 1]
    skel_in[:, 3, 1] = -unnorm_bones[:, 0, 2]
    skel_in[:, 6, 1] = -unnorm_bones[:, 0, 2]
    skel_in[:, 7, 1] = unnorm_bones[:, 0, 3]
    skel_in[:, 8, 1] = unnorm_bones[:, 0, 4]
    skel_in[:, 9, 1] = unnorm_bones[:, 0, 5]
    skel_in[:, 10, 1] = unnorm_bones[:, 0, 6]
    skel_in[:, 11, 0] = unnorm_bones[:, 0, 7]
    skel_in[:, 12, 0] = unnorm_bones[:, 0, 8]
    skel_in[:, 13, 0] = unnorm_bones[:, 0, 9]
    skel_in[:, 14, 0] = -unnorm_bones[:, 0, 7]
    skel_in[:, 15, 0] = -unnorm_bones[:, 0, 8]
    skel_in[:, 16, 0] = -unnorm_bones[:, 0, 9]
    return skel_in


