import torch
import argparse
import time

from data_proc import load_data
from models import *
import torch_geometric.transforms as T

def build_model(args, num_features, num_classes):
    if args.model == 'pgnn':
        model = pGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout)
    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'sgc':
        model = SGCNet(in_channels=num_features,
                        out_channels=num_classes,
                        K=args.K)
    elif args.model == 'gat':
        model = GATNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        num_heads=args.num_heads,
                        dropout=args.dropout)
    elif args.mode