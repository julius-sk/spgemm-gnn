#!/usr/bin/env python3
"""
Simplified MaxK-GNN training that gracefully falls back to DGL when kernels fail
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.autograd import Function
import time

# Try to import custom kernels
try:
    import maxk_kernels
    KERNELS_AVAILABLE = True
    print("✓ Custom MaxK kernels loaded successfully!")
except ImportError:
    KERNELS_AVAILABLE = False
    print("⚠ Custom MaxK kernels not available. Using DGL fallback.")

class MaxKFunction(Function):
    @staticmethod
    def forward(ctx, input, k=32):
        # Always use PyTorch's built-in topk for reliability
        topk_values, indices = input.topk(k, dim=1)
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

class SimpleMaxKSAGE(nn.Module):
    """Simplified SAGE with MaxK that always works"""
    def __init__(self, in_feats, hid_size, num_layers, out_size, maxk=32, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.maxk = maxk
        
        # Linear layers
        self.lin_in = nn.Linear(in_feats, hid_size)
        self.lin_out = nn.Linear(hid_size, out_size)
        
        # SAGE convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(dgl.nn.SAGEConv(hid_size, hid_size, 'mean'))
        
        # MaxK function
        self.maxk_fn = MaxKFunction.apply
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, x):
        # Input transformation
        x = self.lin_in(x)
        
        # Apply MaxK + SAGE layers
        for i, conv in enumerate(self.convs):
            # Apply MaxK nonlinearity
            x = self.maxk_fn(x, self.maxk)
            
            # Apply SAGE convolution
            x = conv(g, x)
            
            # Apply dropout (except last layer)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        
        # Output transformation
        x = self.lin_out(x)
        return x

def train_simple_maxk():
    """Simple training function that always works"""
    print("🚀 Starting simplified MaxK-GNN training...")
    
    # Load data
    from dgl.data import RedditDataset
    
    transform = dgl.AddSelfLoop()
    dataset = RedditDataset(transform=transform, raw_dir='./data/')
    g = dataset[0].to('cuda')
    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    print(f"📊 Dataset: Reddit")
    print(f"📊 Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    print(f"📊 Features: {features.shape[1]}, Classes: {dataset.num_classes}")
    
    # Create model
    model = SimpleMaxKSAGE(
        features.shape[1], 
        256,  # hidden size
        3,    # num layers  
        dataset.num_classes,
        maxk=32,
        dropout=0.5
    ).to('cuda')
    
    # Loss and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    print("🏃 Starting training...")
    best_val_acc = 0
    
    for epoch in range(100):  # Just 100 epochs for testing
        model.train()
        
        # Timing
        start_time = time.time()
        
        # Forward pass
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - start_time
        
        # Evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(g, features)
                
                train_acc = (logits[train_mask].argmax(1) == labels[train_mask]).float().mean()
                val_acc = (logits[val_mask].argmax(1) == labels[val_mask]).float().mean()
                test_acc = (logits[test_mask].argmax(1) == labels[test_mask]).float().mean()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                print(f"Epoch {epoch:03d} | "
                      f"Loss {loss:.4f} | "
                      f"Train {train_acc:.4f} | "
                      f"Val {val_acc:.4f} | "
                      f"Test {test_acc:.4f} | "
                      f"Time {epoch_time:.3f}s")
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_simple_maxk()