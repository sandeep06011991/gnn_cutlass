
import torch

import triton
import triton.language as tl

@triton.jit
def spmm_kernel(indptr: torch.tensor, indices: torch.tensor,\
                 output_t:torch.tensor, input_t: torch.tensor, num_nodes:int, 
                    feat_size: tl.constexpr, BLOCK_SIZE: tl.constexpr ):
    node_id = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    # Assumption feat.shape[1] < BLOCK_SIZE
    while(node_id < num_nodes):
        start = tl.load(indptr + node_id)
        end = tl.load(indptr + node_id + 1)
        t = tl.zeros((BLOCK_SIZE,), tl.float32)
        feat_offsets = tl.arange(0,BLOCK_SIZE)
        mask = feat_offsets < feat_size
        for off in range(start, end):
            edge = tl.load(indices + off)
            
            t += tl.load(input_t + edge * feat_size + feat_offsets, mask=mask)
        tl.store(output_t + node_id * feat_size + feat_offsets, t, mask = mask )
        node_id += num_blocks


def spmm(indptr: torch.tensor, \
            indices: torch.tensor, \
                num_nodes: int, \
                  feat: torch.tensor):
    out = torch.empty_like(feat)
    spmm_kernel[num_nodes,](indptr, indices, out, feat, num_nodes=num_nodes, feat_size=feat.shape[1], BLOCK_SIZE=128)
    return out 

if __name__ == "__main__":
        device = 0
        indptr = torch.tensor([0,2,3]).to(device)
        indices = torch.tensor([1,2,3]).to(device)
        feat = torch.ones(4, 128).to(device)
        num_nodes = 2
        out = spmm(indptr=indptr, indices = indices, num_nodes=num_nodes, feat = feat)
        # print(out)    