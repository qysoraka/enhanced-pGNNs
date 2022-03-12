
import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull
from torch_geometric.utils import num_nodes
from cSBM_dataset import dataset_ContextualSBM

def generate_split(data, num_classes, seed=2021, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask

def generate_random_edges(data, random_rate=None, seed=2021):
    np.random.seed(seed)
    n_nodes = num_nodes.maybe_num_nodes(data.edge_index)
    if random_rate == 0:
        return data.edge_index
    elif random_rate == 1:
        num_new_edges = len(data.edge_index.T)
        rd_edge_index_1 = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        rd_edge_index_2 = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        new_edge_index = np.concatenate([rd_edge_index_1.T, rd_edge_index_2.T])
        new_edge_index = list(set([tuple(e_index) for e_index in new_edge_index]))
        new_edge_index = [list(v) for v in new_edge_index]
        new_edge_index = new_edge_index[:num_new_edges]
        new_edge_index.sort()
        new_edge_index = torch.LongTensor(new_edge_index)
        return new_edge_index.T
    else:
        num_new_edges = int(random_rate * len(data.edge_index.T))
        rd_edge_index = np.random.randint(n_nodes-1, size=(2, num_new_edges))