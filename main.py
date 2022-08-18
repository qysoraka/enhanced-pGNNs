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
                       