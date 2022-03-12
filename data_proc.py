
import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull
from torch_geometric.utils import num_nodes
from cSBM_dataset import dataset_ContextualSBM