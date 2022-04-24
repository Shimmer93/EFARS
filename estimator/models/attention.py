import sys
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from models.sem_gcn import SemGCN
from models.mlp import MLP

# In: B x T x 17 x 2
# Out: B x T x 17 x 3
# B: Batch size, T: Length of Time Sequence
class GCNTransformerModel(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(GCNTransformerModel, self).__init__()
        self.gcn = SemGCN(adj, hid_dim, coords_dim, num_layers, nodes_group, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layers, 6)
        
        self.fc1 = nn.Linear(17 * hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 51)
        self.hid_dim = hid_dim

        self.init_params()

    def forward(self, xs):
        b, t, n, _ = xs.shape
        ys = []
        for i in range(t):
            y = self.gcn(xs[:,i,:,:])
            y = self.fc1(y.reshape(b, n * self.hid_dim))
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        ys = self.encoder(ys)
        out = self.fc2(ys)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

# In: B x T x 17 x 2
# Out: B x T x 17 x 3
# B: Batch size, T: Length of Time Sequence
class MLPTransformerModel(nn.Module):
    def __init__(self, hid_dim):
        super(MLPTransformerModel, self).__init__()
        self.mlp = MLP()
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layers, 6)
        
        self.fc1 = nn.Linear(17 * hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 51)
        self.hid_dim = hid_dim

        self.init_params()

    def forward(self, xs):
        b, t, n, _ = xs.shape
        ys = []
        for i in range(t):
            y = self.mlp(xs[:,i,:,:])
            y = self.fc1(y.reshape(b, n * self.hid_dim))
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        ys = self.encoder(ys)
        out = self.fc2(ys)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

# In: B x T x 17 x 2
# Out: B x T x 17 x 3
# B: Batch size, T: Length of Time Sequence
class PureTransformerModel(nn.Module):
    def __init__(self, hid_dim):
        super(PureTransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layers, 6)
        
        self.fc1 = nn.Linear(34, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 51)
        self.hid_dim = hid_dim

        self.init_params()

    def forward(self, xs):
        b, t, n, _ = xs.shape
        ys = xs.reshape(b * t, n * 2)
        ys = self.fc1(ys)
        ys = ys.reshape(b, t, self.hid_dim)
        ys = self.encoder(ys)
        out = self.fc2(ys)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)