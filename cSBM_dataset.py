
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.

"""
This is a script for contexual SBM model and its dataset generator.
contains functions:
        ContextualSBM
        parameterized_Lambda_and_mu
        save_data_to_pickle
    class:
        dataset_ContextualSBM

"""
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
from datetime import datetime
import os.path as osp
import os
import ipdb
import argparse

import torch
from torch_geometric.data import InMemoryDataset



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def ContextualSBM(n, d, Lambda, p, mu, train_percent=0.025, val_percent=0.025):
    # n = 800 #number of nodes
    # d = 5 # average degree
    # Lambda = 1 # parameters
    # p = 1000 # feature dim
    # mu = 1 # mean of Gaussian
    gamma = n/p

    c_in = d + np.sqrt(d)*Lambda
    c_out = d - np.sqrt(d)*Lambda
    y = np.ones(n)
    y[int(n/2)+1:] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i]*y[j] > 0:
                Flip = np.random.binomial(1, c_in/n)
            else:
                Flip = np.random.binomial(1, c_out/n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1/np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu/n)*y[i]*u + Z/np.sqrt(p)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))
    # order edge list and remove duplicates if any.
    data.coalesce()

    num_class = len(np.unique(y))
    val_lb = int(n * val_percent)
    percls_trn = int(round(train_percent * n / num_class))