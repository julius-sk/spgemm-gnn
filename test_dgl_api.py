#!/usr/bin/env python3
"""
Quick test to understand DGL's adjacency matrix API
"""

import dgl
import torch
from dgl.data import RedditDataset

print("Testing DGL adjacency matrix API...")

# Create a small test graph
print("\n1. Testing with small graph:")
g_small = dgl.graph(([0, 1, 2, 0], [1, 2, 0, 2])).to('cuda')
print(f"Small graph: {g_small.num_nodes()} nodes, {g_small.num_edges()} edges")

try:
    result = g_small.adj_tensors('csr')
    print(f"adj_tensors('csr') returned {len(result)} items:")
    for i, item in enumerate(result):
        print(f"  Item {i}: shape={item.shape}, dtype={item.dtype}")
        if len(item) < 20:  # Only print if small
            print(f"    Values: {item}")
except Exception as e:
    print(f"adj_tensors failed: {e}")

# Test other methods
try:
    adj_scipy = g_small.adjacency_matrix_scipy(fmt='csr')
    print(f"\nadjacency_matrix_scipy: shape={adj_scipy.shape}")
    print(f"  indptr: {adj_scipy.indptr}")
    print(f"  indices: {adj_scipy.indices}")
except Exception as e:
    print(f"adjacency_matrix_scipy failed: {e}")

# Test with actual Reddit data
print("\n2. Testing with Reddit dataset:")
try:
    transform = dgl.AddSelfLoop()
    data = RedditDataset(transform=transform, raw_dir='./data/')
    g_reddit = data[0].to('cuda')
    print(f"Reddit graph: {g_reddit.num_nodes()} nodes, {g_reddit.num_edges()} edges")
    
    result = g_reddit.adj_tensors('csr')
    print(f"Reddit adj_tensors('csr') returned {len(result)} items:")
    for i, item in enumerate(result):
        print(f"  Item {i}: shape={item.shape}, dtype={item.dtype}")
        
except Exception as e:
    print(f"Reddit test failed: {e}")

print("\nDGL version info:")
print(f"DGL version: {dgl.__version__}")

# Test what happens with different formats
print("\n3. Testing different formats:")
formats = ['csr', 'csc', 'coo']
for fmt in formats:
    try:
        result = g_small.adj_tensors(fmt)
        print(f"{fmt} format: {len(result)} items")
    except Exception as e:
        print(f"{fmt} format failed: {e}")