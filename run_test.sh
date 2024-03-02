#!/bin/bash

function run_exp(){
    python main.py --input $1 \
                        --train_rate $2 \
                        --val_rate $3 \
                        --model $4 \
                        --num_hid $5 \
                        --lr $6 \
                        --epochs $7 \
                        --weight_decay $8 \
                        --dropout $9 \
                        --mu ${10} \
                        --p ${11} \
                        --K ${12} \
                        --alpha ${13} \
                        --dprate ${14} \
                        --runs ${15}
}

run_exp cora 0.