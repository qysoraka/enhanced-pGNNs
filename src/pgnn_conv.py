
from typing import Optional, Tuple

from torch._C import BenchmarkExecutionStats
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

from torch_geometric.utils import num_nodes
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
