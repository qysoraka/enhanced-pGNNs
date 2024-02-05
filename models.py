
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, APPNP, JumpingKnowledge
from src.pgnn_conv import pGNNConv
from src.gpr_conv import GPR_prop


class pGNNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_hid=16, 
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=True):
        super(pGNNNet, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)        
        return F.log_softmax(x, dim=1)


class MLPNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5):
        super(MLPNet, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)


class GCNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5,
                 cached=True):
        super(GCNNet, self).__init__()