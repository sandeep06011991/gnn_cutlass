
import torch 
from torch_geometric.utils import spmm
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

data = dataset[0]
x = data.x.to(0) 
adj_t = data.adj_t.to(0)
# data['feat']
# out = self.propagate(edge_index, x=x, size=size)
print(adj_t)
e1 = torch.cuda.Event(enable_timing=True)
e2 = torch.cuda.Event(enable_timing=True)
for _ in range(10):
    e1.record()
    spmm(adj_t, x, reduce='sum')
    e2.record()
    e2.synchronize()
    print(e1.elapsed_time(e2)/1000, "time")