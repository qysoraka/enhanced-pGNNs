#! /bin/bash
#
# create_cSBM_dataset.sh

python cSBM_dataset.py --phi -1.0 \
    --name csbm_phi_dense_-1 \
    --root ./dataset \
    --num_nodes 5000 \
    --num_features 2000 \
    --avg_degree 5 \
    --epsilon 3.25 \
    --train_percent 0.6 