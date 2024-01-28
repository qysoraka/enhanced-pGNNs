
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