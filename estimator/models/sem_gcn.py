# Credit: https://github.com/garyzhao/SemGCN/tree/master/models

from __future__ import absolute_import

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class _NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=1, bn_layer=True):
        super(_NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        assert self.inter_channels > 0

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        else:
            raise Exception('Error feature dimension.')

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample > 1:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample))

    def forward(self, x):
        batch_size = x.size(0)  # x: (b, c, t, h, w)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)
        phi_x = phi_x.expand(-1, -1, h, -1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class GraphNonLocal(_NonLocalBlock):
    def __init__(self, in_channels, inter_channels=None, sub_sample=1, bn_layer=True):
        super(GraphNonLocal, self).__init__(in_channels, inter_channels=inter_channels, dimension=1,
                                            sub_sample=sub_sample, bn_layer=bn_layer)

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
        else:
            group_size = len(nodes_group[0])
            assert group_size > 1

            grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
            restored_order = [0] * len(grouped_order)
            for i in range(len(restored_order)):
                for j in range(len(grouped_order)):
                    if grouped_order[j] == i:
                        restored_order[i] = j
                        break

            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj)

    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out

class GCNLSTM(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(GCNLSTM, self).__init__()
        self.gcn = SemGCN(adj, hid_dim, coords_dim, num_layers, nodes_group, p_dropout)
        self.lstm = nn.LSTM(input_size=51, hidden_size=hid_dim, num_layers=3, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hid_dim * 2, hid_dim * 2)
        self.fc2 = nn.Linear(hid_dim * 2, 51)
        self.hid_dim = hid_dim

        self.init_params()

    def forward(self, xs):
        b, t, n, _ = xs.shape
        ys = []
        for i in range(t):
            y = self.gcn(xs[:,i,:,:])
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        ys = ys.reshape(b, t, n * 3)
        out, _ = self.lstm(ys)
        out = out.reshape(b * t, self.hid_dim * 2)
        out = self.fc1(out).relu()
        out = self.fc2(out)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        for _, param in self.lstm.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

        for _, param in self.fc1.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

        for _, param in self.fc2.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
 

class GCNTransformer(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(GCNTransformer, self).__init__()
        self.gcn = SemGCN(adj, hid_dim, coords_dim, num_layers, nodes_group, p_dropout)
        self.transformer = nn.Transformer(d_model=hid_dim, nhead=16, batch_first=True)
        self.fc1 = nn.Linear(51, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 51)
        #self.pos_enc = PositionalEncoding(hid_dim)
        #self.pos_dec = PositionalEncoding(hid_dim)
        self.hid_dim = hid_dim
        mask = self.transformer.generate_square_subsequent_mask(5)
        self.register_buffer('mask', mask)

        self.init_params()

    def forward(self, xs, ts):
        b, t, n, _ = xs.shape
        ys = []
        for i in range(t):
            y = self.gcn(xs[:,i,:,:])
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        ys = ys.reshape(b * t, n * 3)
        hts = ts.reshape(b * t, n * 3)
        ys = self.fc1(ys)
        hts = self.fc1(hts)
        ys = ys.reshape(b, t, self.hid_dim)
        hts = hts.reshape(b, t, self.hid_dim)
        #ys = self.pos_enc(ys)
        #ts = self.pos_dec(ts)
        out = self.transformer(ys, hts, tgt_mask = self.mask)
        out = self.fc2(out)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        for _, param in self.transformer.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

class PureTransformer(nn.Module):
    def __init__(self, hid_dim):
        super(PureTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=hid_dim, nhead=16, batch_first=True)
        self.fc1 = nn.Linear(34, hid_dim)
        self.fc2 = nn.Linear(51, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 51)
        #self.pos_enc = PositionalEncoding(hid_dim)
        #self.pos_dec = PositionalEncoding(hid_dim)
        self.hid_dim = hid_dim
        mask = self.transformer.generate_square_subsequent_mask(5)
        self.register_buffer('mask', mask)

        self.init_params()

    def forward(self, xs, ts):
        b, t, n, _ = xs.shape
        hxs = xs.reshape(b * t, n * 2)
        hts = ts.reshape(b * t, n * 3)
        hxs = self.fc1(hxs)
        hts = self.fc2(hts)
        hxs = hxs.reshape(b, t, self.hid_dim)
        hts = hts.reshape(b, t, self.hid_dim)
        #ys = self.pos_enc(ys)
        #ts = self.pos_dec(ts)
        out = self.transformer(hxs, hts, tgt_mask = self.mask)
        out = self.fc3(out)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc3.bias)
        nn.init.uniform_(self.fc3.weight, -0.1, 0.1)
        for _, param in self.transformer.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)
                
class GCNTransformerModel(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(GCNTransformerModel, self).__init__()
        self.gcn = SemGCN(adj, hid_dim, coords_dim, num_layers, nodes_group, p_dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, 6)
        
        self.fc1 = nn.Linear(51, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 51)
        self.pos_enc = PositionalEncoding(hid_dim)
        self.hid_dim = hid_dim

        self.init_params()

    def forward(self, xs):
        b, t, n, _ = xs.shape
        ys = []
        for i in range(t):
            y = self.gcn(xs[:,i,:,:])
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        ys = ys.reshape(b * t, n * 3)
        ys = self.fc1(ys)
        ys = ys.reshape(b, t, self.hid_dim) * math.sqrt(self.hid_dim)
        ys = self.pos_enc(ys)
        ys = self.encoder(ys)
        out = self.fc2(ys)
        out = out.reshape(b, t, n, 3)
        return out

    def init_params(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc1.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -0.1, 0.1)

if __name__ == '__main__':
    import torch
    import sys
    sys.path.append('../..')
    from utils.graph import adj_mx_from_edges
    from data.human36m import Human36MMetadata

    #m = GCNLSTM(adj=adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False), hid_dim=128)
    #m = PureTransformer(128)
    m = GCNTransformerModel(adj=adj_mx_from_edges(Human36MMetadata.num_joints, Human36MMetadata.skeleton_edges, sparse=False), hid_dim=128)
    #edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, Human36MMetadata.num_joints_orig)), Human36MMetadata.parents)))[:15]
    #print(len(edges))
    #m = SemGCN(adj=adj_mx_from_edges(16, edges, sparse=False), hid_dim=128, nodes_group=Human36MMetadata.skeleton_joints_group)
    x = torch.randn(2,5,17,2)
    #t = torch.randn(2,5,17,3)
    y = m(x)
    print(y.shape)
    