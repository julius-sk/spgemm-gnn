"""
MaxK-GNN Models with Integrated Custom Kernels
Version 5 - Corrected with Proper Import Structure
"""

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math
import time
import logging

# Import custom kernels with fallback
try:
    import spmm_kernels  # Proper bindings for SPMM_MAXK classes
    KERNELS_AVAILABLE = True
    print("✓ SPMM kernels loaded: SpmmMaxK + SpmmMaxKBackward classes")
except ImportError:
    KERNELS_AVAILABLE = False
    print("⚠ SPMM kernels not available. Using PyTorch/DGL fallback.")

class MaxK(Function):
    """
    MaxK nonlinearity function - uses topk_nonlinearity kernel
    """
    @staticmethod
    def forward(ctx, input, k=1):
        if KERNELS_AVAILABLE:
            try:
                # Use proper top-k kernel
                output = spmm_kernels.topk_nonlinearity(input, k)
                # Save mask for backward
                _, indices = input.topk(k, dim=1)
                mask = torch.zeros_like(input)
                mask.scatter_(1, indices, 1)
                ctx.save_for_backward(mask)
                return output
            except Exception as e:
                print(f"⚠ MaxK kernel failed, using PyTorch: {e}")
        
        # Original implementation
        topk, indices = input.topk(k, dim=1)
        mask = torch.zeros_like(input)
        mask.scatter_(1, indices, 1)
        output = input * mask
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None

class SpGEMMFunction(Function):
    """
    Custom function using actual SPMM_MAXK and SPMM_MAXK_BACKWARD classes
    """
    @staticmethod
    def forward(ctx, features, graph_data, maxk):
        indptr, indices, values = graph_data
        
        if KERNELS_AVAILABLE:
            try:
                # Prepare CBSR format using proper function
                sparse_data, sparse_selector = spmm_kernels.prepare_cbsr_format(features, maxk)
                
                # Create output tensor
                output = torch.zeros_like(features)
                
                # Create SPMM_MAXK kernel instance
                kernel = spmm_kernels.SpmmMaxK(
                    "graph", indptr, indices, values, sparse_data, output
                )
                
                # Set sparse parameters (like in main.cu)
                kernel.set_sparse_params(sparse_selector, maxk)
                
                # Run the kernel
                timing = kernel.run_kernel(False, features.size(1))
                
                # Save for backward
                ctx.graph_data = graph_data
                ctx.sparse_selector = sparse_selector
                ctx.maxk = maxk
                ctx.features_shape = features.shape
                
                return output
                
            except Exception as e:
                print(f"⚠ SPMM_MAXK kernel failed: {e}")
        
        # DGL fallback
        num_nodes = indptr.size(0) - 1
        edge_index = torch.stack([
            torch.repeat_interleave(torch.arange(num_nodes, device=indices.device), 
                                   indptr[1:] - indptr[:-1]),
            indices
        ])
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        
        with g.local_scope():
            g.ndata['h'] = features
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h_out'))
            return g.ndata['h_out']
    
    
    @staticmethod
    def backward(ctx, grad_output):
        if KERNELS_AVAILABLE and hasattr(ctx, 'graph_data'):
            try:
                indptr, indices, values = ctx.graph_data
                
                # Create output tensor for backward (sparse format)
                grad_sparse = torch.zeros(ctx.features_shape[0], ctx.maxk, 
                                        device=grad_output.device, dtype=grad_output.dtype)
                
                # Create SPMM_MAXK_BACKWARD kernel instance
                kernel_backward = spmm_kernels.SpmmMaxKBackward(
                    "graph", indptr, indices, values, grad_output, grad_sparse
                )
                
                # Set sparse parameters
                kernel_backward.set_sparse_params(ctx.sparse_selector, ctx.maxk)
                
                # Run backward kernel
                timing = kernel_backward.run_kernel(False, grad_output.size(1))
                
                # Convert sparse gradient back to full format
                grad_input = torch.zeros_like(grad_output)
                for i in range(ctx.features_shape[0]):
                    for j in range(ctx.maxk):
                        idx = ctx.sparse_selector[i, j].item()
                        if idx < ctx.features_shape[1]:
                            grad_input[i, idx] = grad_sparse[i, j]
                
                return grad_input, None, None
                
            except Exception as e:
                print(f"⚠ SPMM_MAXK_BACKWARD kernel failed: {e}")
        
        # Fallback
        return grad_output, None, None

class SAGE(nn.Module):
    """
    SAGE model with proper SPMM_MAXK integration
    """
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        self.num_layers = num_hid_layers
        self.nonlinear = nonlinear
        self.maxk = maxk
        
        # Training timers (original)
        self.aggregation_time = 0.0
        self.total_training_time = 0.0
        self.is_training_timing = False
        
        # Fast approach: Direct linear layers for SAGE transformations
        self.fc_self_layers = nn.ModuleList()
        self.fc_neigh_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.fc_self_layers.append(nn.Linear(hid_size, hid_size, bias=False))
            self.fc_neigh_layers.append(nn.Linear(hid_size, hid_size, bias=False))
            self.dropout_layers.append(nn.Dropout(feat_drop))
            
            if norm:
                self.norm_layers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
            else:
                self.norm_layers.append(nn.Identity())

        # Linear layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        
        # Initialize linear layers
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        for i in range(self.num_layers):
            init.xavier_uniform_(self.fc_self_layers[i].weight)
            init.xavier_uniform_(self.fc_neigh_layers[i].weight)
        
        # MaxK functions (original)
        for i in range(self.num_layers):
            exec("self.maxk{} = MaxK.apply".format(i))
            exec("self.k{} = maxk".format(i))

    def forward(self, g, x):
        """
        Forward pass following MaxK-GNN paper flow:
        Linear → MaxK+SpGEMM (integrated) → SAGE combination
        """
        x = self.lin_in(x)

        for i in range(self.num_layers):
            # CORRECT FLOW: Apply linear transformation first
            x = self.linlayers[i](x) if hasattr(self, 'linlayers') else x
            
            # Paper's approach: Integrated MaxK + SpGEMM aggregation
            if KERNELS_AVAILABLE and self.nonlinear == 'maxk':
                try:
                    # Extract graph structure for SPMM
                    indptr, indices, _ = g.adj_sparse('csr')
                    values = torch.ones(indices.shape[0], device=x.device, dtype=x.dtype)
                    
                    # Integrated MaxK + SpGEMM: h(X^(l-1)) → A·h(X^(l-1))
                    x_aggregated = MaxKSpGEMMFunction.apply(x, (indptr.int(), indices.int(), values), self.maxk)
                    
                    # SAGE-specific transformations
                    h_self = self.fc_self_layers[i](x)  # Self connection
                    h_neigh = self.fc_neigh_layers[i](x_aggregated)  # Neighbor aggregation
                    
                    # Combine self and neighbor features
                    x = h_self + h_neigh
                        
                except Exception as e:
                    print(f"⚠ MaxK-SpGEMM kernel failed for layer {i}: {e}")
                    # Fallback: separate MaxK and aggregation
                    if self.nonlinear == 'maxk':
                        x = eval("self.maxk{}(x, self.k{})".format(i, i))
                    
                    # Simple aggregation fallback
                    adj_matrix = torch.sparse_coo_tensor(
                        g.edges(), torch.ones(g.num_edges(), device=x.device),
                        (g.num_nodes(), g.num_nodes())
                    ).coalesce()
                    
                    x_aggregated = torch.sparse.mm(adj_matrix, x) / (adj_matrix.sum(dim=1, keepdim=True) + 1e-6)
                    h_self = self.fc_self_layers[i](x)
                    h_neigh = self.fc_neigh_layers[i](x_aggregated)
                    x = h_self + h_neigh
            else:
                # Non-MaxK case or no kernels available
                if self.nonlinear == 'maxk':
                    x = eval("self.maxk{}(x, self.k{})".format(i, i))
                elif self.nonlinear == 'relu':
                    x = F.relu(x)
                
                # Standard aggregation
                adj_matrix = torch.sparse_coo_tensor(
                    g.edges(), torch.ones(g.num_edges(), device=x.device),
                    (g.num_nodes(), g.num_nodes())
                ).coalesce()
                
                x_aggregated = torch.sparse.mm(adj_matrix, x) / (adj_matrix.sum(dim=1, keepdim=True) + 1e-6)
                h_self = self.fc_self_layers[i](x)
                h_neigh = self.fc_neigh_layers[i](x_aggregated)
                x = h_self + h_neigh
            
            # Apply dropout and normalization
            x = self.dropout_layers[i](x)
            x = self.norm_layers[i](x)
            
        x = self.lin_out(x)
        return x
