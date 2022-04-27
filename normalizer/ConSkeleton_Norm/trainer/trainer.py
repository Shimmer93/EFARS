import numpy as np
import torch
import random

from model import metric
from base.base_trainer import base_trainer

from utils import h36m_utils, visualization
from utils.logger import Logger

class fk_trainer(base_trainer):
    def __init__(self, model, resume, config, data_loader, test_data_loader):
        super(fk_trainer, self).__init__(model, resume, config, logger_path='%s/%s.log' % (config.trainer.checkpoint_dir, config.trainer.checkpoint_dir.split('/')[-1]))
        self.config = config
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.log_step = 10#int(np.sqrt(data_loader.batch_size))
        self.train_parameters = [self._prepare_data(item) for item in self.data_loader.dataset.get_parameters()]
        self.test_parameters = [self._prepare_data(item) for item in self.test_data_loader.dataset.get_parameters()]
        self.lambda_recon_cycle, self.lambda_rotation_cycle, self.lambda_pee = 0.06, 0.1, 1.2
        self.stpes = 0

    def _train_epoch(self, epoch):
        def get_velocity(motions, joint_index):
            joint_motion = motions[..., [joint_index*3, joint_index*3 + 1, joint_index*3 + 2]]
            velocity = torch.sqrt(torch.sum((joint_motion[:, 2:] - joint_motion[:, :-2])**2, dim=-1))
            return velocity
        self.model.train()
        for batch_idx, datas in enumerate(self.data_loader):
            datas = [self._prepare_data(item, _from='tensor') for item in datas]
            poses_2d, poses_3d, bones, contacts, alphas, proj_facters, cam_para = datas
            fake_rotations, fake_rotations_cycle, src_pose_3d, src_pose_3d_cycle = self.model.forward_fk(poses_2d, bones, self.train_parameters, cam_para)
            position_weights = torch.ones((1, 17)).cuda()
            position_weights[:, [0, 3, 6, 8, 11, 14]] = self.lambda_pee
            loss_recon = torch.mean(torch.norm((src_pose_3d.view((-1, 17, 3)) - poses_3d.view((-1, 17, 3))), dim=-1)*position_weights)
            loss_recon_cycle = torch.mean(torch.norm((src_pose_3d_cycle.view((-1, 17, 3)) - poses_3d.view((-1, 17, 3))), dim=-1)*position_weights)
            loss_rotation_cycle = torch.nn.MSELoss()(fake_rotations, fake_rotations_cycle)
            loss_G = loss_recon + loss_recon_cycle * self.lambda_recon_cycle + loss_rotation_cycle * self.lambda_rotation_cycle
            
            self.model.optimizer_Q.zero_grad()
            loss_G.backward()
            self.model.optimizer_Q.step()


            train_log = {'loss_G': loss_G, 'loss_recon': loss_recon, 'loss_recon_cycle':loss_recon_cycle, 'loss_rotation_cycle':loss_rotation_cycle}

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.writer.set_step(self.stpes, mode='train')
                self.writer.set_scalars(train_log)
                training_message = 'Train Epoch: {} [{}/{} ({:.0f})%]\t'.format(epoch, self.data_loader.batch_size*batch_idx, self.data_loader.n_samples, 100.0 * batch_idx / len(self.data_loader))
                for key, value in train_log.items():
                    if value > 0:
                        training_message += '{}: {:.6f}\t'.format(key, value)
                self.train_logger.info(training_message)
            self.stpes += 1
        
        val_log = self._valid_epoch(epoch)
        self.data_loader.dataset.set_sequences()
        return val_log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_metrics = 0
        total_val_loss = 0
        for batch_idx, datas in enumerate(self.test_data_loader):
            datas = [self._prepare_data(item, _from='tensor') for item in datas[:-1]]
            poses_2d_pixel, poses_2d, poses_3d, bones, contacts, alphas, proj_facters = datas
            _, _, _, fake_pose_3d, _, _ = self.model.forward_fk(poses_2d, self.test_parameters)
            total_val_metrics += metric.mean_points_error(fake_pose_3d, poses_3d) * torch.mean(alphas[0]).data.cpu().numpy()
            total_val_loss += torch.mean(torch.norm(fake_pose_3d.view(-1, 17, 3) - poses_3d.view(-1, 17, 3), dim=-1)).item()
        val_log = {'val_metric': total_val_metrics/len(self.test_data_loader), 'val_loss': total_val_loss/len(self.test_data_loader),}
        self.writer.set_step(epoch, mode='valid')
        self.writer.set_scalars(val_log)
        self.train_logger.info('Eveluation: mean_points_error: {:.6f} loss: {:.6f}'.format(val_log['val_metric'], val_log['val_loss']))
        return val_log