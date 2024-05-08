# Enhanced pGNNs
This repository provides a refined version of pGNN as described in the ICML'2022 published paper '**[p-Laplacian Based Graph Neural Networks](https://proceedings.mlr.press/v162/fu22e.html)**'.

### Requirements
The following packages are required: 

- [pytorch 1.9.0](https://pytorch.org/get-started/locally/)
- [torch_geometric 1.7.1](https://github.com/pyg-team/pytorch_geometric)
- scikit-learn 0.24.2
- networkx 2.5.1

### Basic Usage
Run the following command:
```
$ python main.py --input cora --train_rate 0.025 --val_rate 0.025 --model pgnn --mu 0.1  --p 2 --K 4 --num_hid 16 --lr 0.01 --epochs 1000 
```

### Testing Examples
Run the following command:
```
$ bash run_test.sh
```

### Citing
If you find the pGNN model useful in your research, please con