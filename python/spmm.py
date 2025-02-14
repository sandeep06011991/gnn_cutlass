
import torch

import triton
import triton.language as tl

@triton.jit
def spmm_kernel(indptr, indices, out, n_elements, BLOCK_SIZE: tl.constexpr ):
      
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):

def spmm(indptr: torch.tensor, \
            indices: torch.tensor, \
                num_nodes: int, num_edges: int, \
                  feat: torch.tensor):
    pass 


if __name__ == "__main__":
        spmm()
    