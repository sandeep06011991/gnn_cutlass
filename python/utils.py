
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def get_pyg_dataset(dataset: str):
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = '/data/sandeep/datasets'
                                     transform=T.ToSparseTensor())