import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, edge_index, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, 3)

        self.edge_index = edge_index

    def forward(self, x: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, self.edge_index).relu()
        res = x
        x = self.conv2(x, self.edge_index)
        x = (self.conv3(x, self.edge_index) + res).relu()
        x = self.conv4(x, self.edge_index)
        return x

#model = GCN(2, 16, 3)