# Credit: https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/lib/models/posenet.py

import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        print(f"Fetch model weight from {load_dir}")
        checkpoint = torch.load(load_dir, map_location='cpu')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.batch_norm1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w1(y)

        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)

        out = x + y

        return out

# In: B x 17 x 2
# Out: B x 17 x 3
# B: Batch size
class MLP(nn.Module):
    def __init__(self,
                 num_joint,
                 linear_size=4096,
                 num_stage=2,
                 p_dropout=0.5,
                 pretrained=False):
        super(MLP, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  num_joint * 2
        # 3d joints
        self.output_size = num_joint * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # pre-processing
        y = self.w1(x.reshape((x.shape[0], -1)))

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y.reshape((y.shape[0], y.shape[1]//3, 3))

    def _load_pretrained_model(self):
        #pass
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir='/home/zpengac/pose/EFARS/estimator/best.pth.tar', pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])