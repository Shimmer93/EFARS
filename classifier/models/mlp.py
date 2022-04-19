# Credit: https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/lib/models/posenet.py

import torch
import torch.nn as nn
#from core.config import cfg as cfg
#from funcs_utils import load_checkpoint


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


class MLP(nn.Module):
    def __init__(self,
                 num_joint,
                 num_classes,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5,
                 pretrained=False):
        super(MLP, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.input_size =  num_joint * 3
        self.output_size = num_classes

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

        return y

    def _load_pretrained_model(self):
        #pass
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir='/home/zpengac/pose/EFARS/estimator/best.pth.tar', pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])

class MLPTransformerModel(nn.Module):
    def __init__(self, hid_dim, num_joint, num_classes, seq_len):
        super(MLPTransformerModel, self).__init__()
        self.mlp = MLP(num_joint, hid_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layers, 6)
        
        self.fc1 = nn.Linear(seq_len * hid_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.hid_dim = hid_dim

        self.init_params()

    def forward(self, xs):
        b, t, n, _ = xs.shape
        ys = []
        for i in range(t):
            y = self.mlp(xs[:,i,:,:])
            #y = self.fc1(y.reshape(b, n * self.hid_dim))
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        ys = self.encoder(ys)
        ys = ys.reshape(b, -1)
        out = self.fc1(ys).relu()
        out = self.fc2(out)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

if __name__ == '__main__':
    import torch
    #m = MLP(num_joint=17, num_classes=11, pretrained=False)
    #x = torch.randn(2,17,3)
    m = MLPTransformerModel(128, 17, 11, 8)
    x = torch.randn(2, 8, 17, 3)
    y = m(x)
    print(y.shape)