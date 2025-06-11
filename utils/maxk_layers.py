import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import dgl
import dgl.function as fn

# Import our custom kernels
try:
    import maxk_kernels
    KERNELS_AVAILABLE = True
except ImportError:
    print("Warning: Custom MaxK kernels not available. Using fallback implementation.")
    KERNELS_AVAILABLE = False

class MaxKFunction(Function):
    @staticmethod
    def forward(ctx, input, k=32):
        if KERNELS_AVAILABLE:
            # Use custom CUDA kernel
            output = maxk_kernels.maxk_forward(input, k)
            # Extract indices for backward pass
            indices = torch.topk(input, k, dim=1)[1]
            ctx.save_for_backward(indices)
            ctx.k = k
            return output
        else:
            # Fallback implementation
            topk_values, indices = input.topk(k, dim=1)
            mask = torch.zeros_like(input)
            mask.scatter_(1, indices, 1)
            output = input * mask
            ctx.save_for_backward(mask)
            return output

    @staticmethod
    def backward(ctx, grad_output):
        if KERNELS_AVAILABLE:
            indices, = ctx.saved_tensors
            grad_input = maxk_kernels.maxk_backward(grad_output, indices)
            return grad_input, None
        else:
            mask, = ctx.saved_tensors
            grad_input = grad_output * mask
            return grad_input, None

class MaxKSAGEConv(nn.Module):
    """
    Custom SAGEConv layer using MaxK nonlinearity and custom SpGEMM kernels
    """
    def __init__(self, in_feats, out_feats, aggregator_type='mean', 
                 feat_drop=0., norm=None, maxk=32):
        super(MaxKSAGEConv, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        self.maxk = maxk
        
        # Linear transformations
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        
        # Normalization
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_feats)
            
        # Dropout
        self.feat_drop = nn.Dropout(feat_drop)
        
        # MaxK function
        self.maxk_fn = MaxKFunction.apply
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            # Apply linear transformation first
            h_self = self.fc_self(feat)
            h_neigh = self.fc_neigh(feat)
            
            # Apply MaxK nonlinearity to create sparse features
            h_neigh_sparse = self.maxk_fn(h_neigh, self.maxk)
            
            # Store sparse features and indices for aggregation
            graph.ndata['h'] = h_neigh_sparse
            
            if KERNELS_AVAILABLE and hasattr(graph, '_sparse_format'):
                # Use custom SpGEMM kernel
                return self._aggregate_with_custom_kernel(graph, h_self, h_neigh_sparse)
            else:
                # Use DGL's built-in aggregation
                return self._aggregate_with_dgl(graph, h_self)
    
    def _aggregate_with_custom_kernel(self, graph, h_self, h_neigh_sparse):
        """Aggregate using custom SpGEMM kernel"""
        # Extract graph structure in CSR format
        try:
            # Try DGL 2.0+ API - returns (indptr, indices, edge_ids)
            result = graph.adj_tensors('csr')
            if len(result) == 3:
                ptr, idx, _ = result  # Ignore edge_ids
            elif len(result) == 2:
                ptr, idx = result
            else:
                raise ValueError(f"Unexpected adj_tensors result: {len(result)} values")
        except (AttributeError, ValueError):
            try:
                # Try using adjacency_matrix_scipy and convert
                import scipy.sparse as sp
                adj_scipy = graph.adjacency_matrix_scipy(fmt='csr')
                ptr = torch.from_numpy(adj_scipy.indptr).to(graph.device)
                idx = torch.from_numpy(adj_scipy.indices).to(graph.device)
            except:
                try:
                    # Try manual edge construction
                    edges = graph.edges()
                    src, dst = edges[0], edges[1]
                    num_nodes = graph.num_nodes()
                    
                    # Create adjacency list manually
                    adj_list = [[] for _ in range(num_nodes)]
                    for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
                        adj_list[s].append(d)
                    
                    # Convert to CSR format
                    ptr = [0]
                    idx = []
                    for neighbors in adj_list:
                        idx.extend(sorted(neighbors))
                        ptr.append(len(idx))
                    
                    ptr = torch.tensor(ptr, dtype=torch.int32, device=graph.device)
                    idx = torch.tensor(idx, dtype=torch.int32, device=graph.device)
                except Exception as e:
                    print(f"Failed to extract CSR format: {e}")
                    # Fall back to DGL aggregation
                    return self._aggregate_with_dgl(graph, h_self)
        
        # Create edge values (for different aggregator types)
        if self.aggregator_type == 'mean':
            degrees = graph.in_degrees().float().clamp(min=1)
            # Create edge weights for mean aggregation
            edge_weights = []
            for i in range(graph.num_nodes()):
                start_idx = ptr[i].item()
                end_idx = ptr[i + 1].item()
                num_neighbors = end_idx - start_idx
                if num_neighbors > 0:
                    edge_weights.extend([1.0 / degrees[i].item()] * num_neighbors)
            edge_weights = torch.tensor(edge_weights, device=graph.device)
        else:
            edge_weights = torch.ones(graph.num_edges(), device=graph.device)
        
        # Extract sparse features (sp_data and sp_index) 
        sp_data, sp_index = self._extract_sparse_format(h_neigh_sparse)
        
        # Apply custom SpGEMM kernel
        try:
            aggregated_feat, _ = maxk_kernels.spgemm_forward(
                ptr.int(), idx.int(), edge_weights,
                sp_data, sp_index,
                graph.num_nodes(), graph.num_edges(),
                self.maxk, self.out_feats
            )
            
            # Combine self and neighbor features
            output = h_self + aggregated_feat
            
        except Exception as e:
            print(f"Custom kernel failed: {e}, falling back to DGL")
            return self._aggregate_with_dgl(graph, h_self)
        
        # Apply normalization and dropout
        if self.norm is not None:
            output = self.norm(output)
        
        return self.feat_drop(output)
    
    def _aggregate_with_dgl(self, graph, h_self):
        """Fallback aggregation using DGL"""
        # Use the original sparse features stored in graph
        if hasattr(graph, 'ndata') and 'h' in graph.ndata:
            if self.aggregator_type == 'mean':
                graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            elif self.aggregator_type == 'sum':
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            else:
                raise ValueError(f"Unsupported aggregator type: {self.aggregator_type}")
            
            h_neigh = graph.ndata['neigh']
            output = h_self + h_neigh
            
            if self.norm is not None:
                output = self.norm(output)
                
            return self.feat_drop(output)
        else:
            raise RuntimeError("Graph data not properly set for DGL fallback")
    
    def _aggregate_with_dgl(self, graph, h_self):
        """Fallback aggregation using DGL"""
        if self.aggregator_type == 'mean':
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
        elif self.aggregator_type == 'sum':
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
        else:
            raise ValueError(f"Unsupported aggregator type: {self.aggregator_type}")
        
        h_neigh = graph.ndata['neigh']
        output = h_self + h_neigh
        
        if self.norm is not None:
            output = self.norm(output)
            
        return self.feat_drop(output)
    
    def _extract_sparse_format(self, sparse_tensor):
        """Extract CBSR format from sparse tensor"""
        # sparse_tensor is the output from MaxK function
        batch_size, orig_dim = sparse_tensor.size()
        
        # Find non-zero elements and their indices
        non_zero_mask = sparse_tensor != 0
        
        # For each row, find the indices of non-zero elements
        sp_data_list = []
        sp_index_list = []
        
        for i in range(batch_size):
            row_mask = non_zero_mask[i]
            non_zero_indices = torch.nonzero(row_mask, as_tuple=True)[0]
            non_zero_values = sparse_tensor[i][non_zero_indices]
            
            # Pad to consistent size (k elements)
            k = min(len(non_zero_indices), self.maxk)
            if k == 0:
                # Handle case with no non-zero elements
                padded_values = torch.zeros(self.maxk, device=sparse_tensor.device)
                padded_indices = torch.zeros(self.maxk, dtype=torch.uint8, device=sparse_tensor.device)
            else:
                # Take top k (they should already be the top k from MaxK)
                if len(non_zero_indices) >= self.maxk:
                    padded_values = non_zero_values[:self.maxk]
                    padded_indices = non_zero_indices[:self.maxk].to(torch.uint8)
                else:
                    # Pad with zeros if we have fewer than k elements
                    padded_values = torch.zeros(self.maxk, device=sparse_tensor.device)
                    padded_indices = torch.zeros(self.maxk, dtype=torch.uint8, device=sparse_tensor.device)
                    padded_values[:k] = non_zero_values
                    padded_indices[:k] = non_zero_indices.to(torch.uint8)
            
            sp_data_list.append(padded_values)
            sp_index_list.append(padded_indices)
        
        sp_data = torch.stack(sp_data_list)
        sp_index = torch.stack(sp_index_list)
        
        return sp_data, sp_index

class MaxKGCNConv(nn.Module):
    """
    Custom GCNConv layer using MaxK nonlinearity and custom SpGEMM kernels
    """
    def __init__(self, in_feats, out_feats, norm='both', weight=True, 
                 bias=True, allow_zero_in_degree=False, maxk=32):
        super(MaxKGCNConv, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.maxk = maxk
        self.allow_zero_in_degree = allow_zero_in_degree
        
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
            
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        
        self.maxk_fn = MaxKFunction.apply
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            if not self.allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise ValueError('Graph has nodes with zero in-degree')
            
            # Linear transformation
            if self.weight is not None:
                feat = torch.mm(feat, self.weight)
            
            # Apply MaxK nonlinearity
            feat_sparse = self.maxk_fn(feat, self.maxk)
            
            # Normalization
            if self.norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                feat_sparse = feat_sparse * norm.unsqueeze(1)
            
            graph.ndata['h'] = feat_sparse
            
            if KERNELS_AVAILABLE:
                return self._aggregate_with_custom_kernel(graph, feat_sparse)
            else:
                return self._aggregate_with_dgl(graph)
    
    def _aggregate_with_custom_kernel(self, graph, feat_sparse):
        """Use custom SpGEMM kernel for aggregation"""
        # Extract graph structure in CSR format
        try:
            # Try DGL 2.0+ API - returns (indptr, indices, edge_ids)
            result = graph.adj_tensors('csr')
            if len(result) == 3:
                ptr, idx, _ = result  # Ignore edge_ids
            elif len(result) == 2:
                ptr, idx = result
            else:
                raise ValueError(f"Unexpected adj_tensors result: {len(result)} values")
        except (AttributeError, ValueError):
            try:
                # Try using adjacency_matrix_scipy and convert
                import scipy.sparse as sp
                adj_scipy = graph.adjacency_matrix_scipy(fmt='csr')
                ptr = torch.from_numpy(adj_scipy.indptr).to(graph.device)
                idx = torch.from_numpy(adj_scipy.indices).to(graph.device)
            except:
                try:
                    # Try manual edge construction
                    edges = graph.edges()
                    src, dst = edges[0], edges[1]
                    num_nodes = graph.num_nodes()
                    
                    # Create adjacency list manually
                    adj_list = [[] for _ in range(num_nodes)]
                    for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
                        adj_list[s].append(d)
                    
                    # Convert to CSR format
                    ptr = [0]
                    idx = []
                    for neighbors in adj_list:
                        idx.extend(sorted(neighbors))
                        ptr.append(len(idx))
                    
                    ptr = torch.tensor(ptr, dtype=torch.int32, device=graph.device)
                    idx = torch.tensor(idx, dtype=torch.int32, device=graph.device)
                except Exception as e:
                    print(f"Failed to extract CSR format: {e}")
                    # Fall back to DGL aggregation
                    return self._aggregate_with_dgl(graph)
        sp_data, sp_index = self._extract_sparse_format(feat_sparse)
        
        # GCN normalization weights
        if self.norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm_right = torch.pow(degs, -0.5)
            edge_weights = norm_right[idx]
        else:
            edge_weights = torch.ones(graph.num_edges(), device=graph.device)
        
        output, _ = maxk_kernels.spgemm_forward(
            ptr, idx, edge_weights,
            sp_data, sp_index,
            graph.num_nodes(), graph.num_edges(),
            self.maxk, self.out_feats
        )
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def _aggregate_with_dgl(self, graph):
        """Fallback using DGL"""
        if self.norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            graph.ndata['h'] = graph.ndata['h'] * norm.unsqueeze(1)
        
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        output = graph.ndata['h']
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def _extract_sparse_format(self, sparse_tensor):
        """Extract CBSR format from sparse tensor"""
        # sparse_tensor is the output from MaxK function
        batch_size, orig_dim = sparse_tensor.size()
        
        # Find non-zero elements and their indices
        non_zero_mask = sparse_tensor != 0
        
        # For each row, find the indices of non-zero elements
        sp_data_list = []
        sp_index_list = []
        
        for i in range(batch_size):
            row_mask = non_zero_mask[i]
            non_zero_indices = torch.nonzero(row_mask, as_tuple=True)[0]
            non_zero_values = sparse_tensor[i][non_zero_indices]
            
            # Pad to consistent size (k elements)
            k = min(len(non_zero_indices), self.maxk)
            if k == 0:
                # Handle case with no non-zero elements
                padded_values = torch.zeros(self.maxk, device=sparse_tensor.device)
                padded_indices = torch.zeros(self.maxk, dtype=torch.uint8, device=sparse_tensor.device)
            else:
                # Take top k (they should already be the top k from MaxK)
                if len(non_zero_indices) >= self.maxk:
                    padded_values = non_zero_values[:self.maxk]
                    padded_indices = non_zero_indices[:self.maxk].to(torch.uint8)
                else:
                    # Pad with zeros if we have fewer than k elements
                    padded_values = torch.zeros(self.maxk, device=sparse_tensor.device)
                    padded_indices = torch.zeros(self.maxk, dtype=torch.uint8, device=sparse_tensor.device)
                    padded_values[:k] = non_zero_values
                    padded_indices[:k] = non_zero_indices.to(torch.uint8)
            
            sp_data_list.append(padded_values)
            sp_index_list.append(padded_indices)
        
        sp_data = torch.stack(sp_data_list)
        sp_index = torch.stack(sp_index_list)
        
        return sp_data, sp_index