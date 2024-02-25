#!/bin/bash

function run_exp(){
    python main.py --input $1 \
                        --train_rate $2 \
                        --val_rate $3 \
    