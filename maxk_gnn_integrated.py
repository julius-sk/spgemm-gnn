import argparse
import os
import time

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset, FlickrDataset, YelpDataset

from utils.config import TrainConfig
import utils.general_utils as general_utils
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from utils.proteins_loader import load_proteins

# Import our integrated models
from utils.integrated_models import MaxKSAGE, MaxKGCN, MaxKGIN

# Try to import custom kernels
try:
    import maxk_kernels
    print("✓ Custom MaxK kernels loaded successfully!")
    KERNELS_AVAILABLE = True
except ImportError:
    print("⚠ Custom MaxK kernels not available. Using fallback implementations.")
    print("To enable custom kernels, run: python setup.py install")
    KERNELS_AVAILABLE = False

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def evaluate(g, features, labels, mask, model, config, logger):
    model.eval()
    if config.dataset == 'ogbn-proteins':
        evaluator = Evaluator(name='ogbn-proteins')
        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred, "y_true": labels}
        )["rocauc"]
    
    with torch.no_grad():
        logits = model(g, features)
        if config.dataset != 'ogbn-proteins':
            return general_utils.compute_micro_f1(logits, labels, mask)
        else:
            return evaluator_wrapper(logits[mask], labels[mask])

def evaluate_masks(g, features, labels, masks, model, config, logger):
    model.eval()
    train_mask = masks[0]
    val_mask = masks[1] 
    test_mask = masks[2]
    
    if config.dataset == 'ogbn-proteins':
        evaluator = Evaluator(name='ogbn-proteins')
        evaluator_wrapper = lambda pred, labels: evaluator.eval(
            {"y_pred": pred, "y_true": labels}
        )["rocauc"]
    
    with torch.no_grad():
        logits = model(g, features)
        if config.dataset != 'ogbn-proteins':
            train_acc = general_utils.compute_micro_f1(logits, labels, train_mask)
            val_acc = general_utils.compute_micro_f1(logits, labels, val_mask)
            test_acc = general_utils.compute_micro_f1(logits, labels, test_mask)
        else:
            train_acc = evaluator_wrapper(logits[train_mask], labels[train_mask])
            val_acc = evaluator_wrapper(logits[val_mask], labels[val_mask])
            test_acc = evaluator_wrapper(logits[test_mask], labels[test_mask])
        
        return train_acc, val_acc, test_acc

def prepare_graph_for_kernels(g):
    """
    Prepare graph data structures for custom kernels
    """
    if not KERNELS_AVAILABLE:
        return g
    
    # Convert to CSR format and cache
    try:
        # Try different DGL API versions
        try:
            # DGL 2.0+ API - may return (indptr, indices, edge_ids)
            result = g.adj_tensors('csr')
            if len(result) == 3:
                ptr, idx, _ = result
            elif len(result) == 2:
                ptr, idx = result
            else:
                raise ValueError(f"Unexpected adj_tensors result: {len(result)} values")
        except (AttributeError, ValueError):
            try:
                # Try using adjacency_matrix_scipy
                import scipy.sparse as sp
                adj_scipy = g.adjacency_matrix_scipy(fmt='csr')
                ptr = torch.from_numpy(adj_scipy.indptr).to(g.device)
                idx = torch.from_numpy(adj_scipy.indices).to(g.device)
            except:
                # Manual construction
                edges = g.edges()
                src, dst = edges[0], edges[1]
                num_nodes = g.num_nodes()
                
                # Create adjacency list
                adj_list = [[] for _ in range(num_nodes)]
                for s, d in zip(src.cpu().numpy(), dst.cpu().numpy()):
                    adj_list[s].append(d)
                
                # Convert to CSR
                ptr = [0]
                idx = []
                for neighbors in adj_list:
                    idx.extend(sorted(neighbors))
                    ptr.append(len(idx))
                
                ptr = torch.tensor(ptr, dtype=torch.int32, device=g.device)
                idx = torch.tensor(idx, dtype=torch.int32, device=g.device)
        
        g._sparse_format = {
            'ptr': ptr,
            'idx': idx,
            'format': 'csr'
        }
        print(f"✓ Graph prepared for custom kernels: {g.num_nodes()} nodes, {g.num_edges()} edges")
    except Exception as e:
        print(f"⚠ Warning: Could not prepare graph for custom kernels: {e}")
        print("Will use DGL fallback for aggregation")
        g._sparse_format = None
    
    return g

def train(g, features, labels, masks, model, config, logger, writer):
    train_mask = masks[0]
    
    # Loss function
    if (config.dataset != 'yelp') and (config.dataset != 'ogbn-proteins'):
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = F.binary_cross_entropy_with_logits
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.w_lr, 
                                weight_decay=config.w_weight_decay)
    if config.enable_lookahead:
        optimizer = general_utils.Lookahead(optimizer)
    
    # Training metrics
    best_val_accuracy = 0
    best_test_accuracy = 0
    total_time = 0
    
    print(f"\n🚀 Starting training with {'custom kernels' if KERNELS_AVAILABLE else 'DGL fallback'}...")
    print(f"Model: {config.model}, Dataset: {config.dataset}, MaxK: {config.maxk}")
    
    for epoch in range(config.epochs):
        model.train()
        
        # Timing for performance measurement
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Forward pass
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        total_time += epoch_time
        
        # Evaluation
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            train_acc, val_acc, test_acc = evaluate_masks(g, features, labels, masks, model, config, logger)
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_test_accuracy = test_acc
            
            # Logging
            writer.add_scalar('train/loss', loss.item(), epoch)
            writer.add_scalar('train/train_acc', train_acc, epoch)
            writer.add_scalar('train/val_acc', val_acc, epoch)
            writer.add_scalar('train/test_acc', test_acc, epoch)
            writer.add_scalar('train/epoch_time', epoch_time, epoch)
            
            logger.info(
                f"Epoch {epoch:04d}/{config.epochs:04d} | "
                f"Loss {loss.item():.4f} | "
                f"Train {train_acc:.4f} | "
                f"Val {val_acc:.4f} | "
                f"Test {test_acc:.4f} | "
                f"Best Val {best_val_accuracy:.4f} | "
                f"Best Test {best_test_accuracy:.4f} | "
                f"Time {epoch_time:.3f}s"
            )
    
    avg_epoch_time = total_time / config.epochs
    logger.info(f"\n📊 Training completed!")
    logger.info(f"Average epoch time: {avg_epoch_time:.3f}s")
    logger.info(f"Total training time: {total_time:.2f}s")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Best test accuracy: {best_test_accuracy:.4f}")
    
    return best_val_accuracy, best_test_accuracy, avg_epoch_time

def main():
    config = TrainConfig()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    # Setup logging
    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)
    
    logger = general_utils.get_logger(os.path.join(config.path, f"{config.dataset}.log"))
    config.print_params(logger.info)
    
    # GPU setup
    torch.cuda.set_device(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"🔧 Using device: {device}")
    logger.info(f"🔧 Custom kernels: {'Available' if KERNELS_AVAILABLE else 'Not available'}")
    
    # Load dataset
    if "ogb" not in config.dataset:
        transform = AddSelfLoop()
        if config.dataset == 'reddit':
            data = RedditDataset(transform=transform, raw_dir=config.data_path)
        elif config.dataset == 'flickr':
            data = FlickrDataset(transform=transform, raw_dir=config.data_path)
        elif config.dataset == 'yelp':
            data = YelpDataset(transform=transform, raw_dir=config.data_path)
        
        g = data[0]
        g = g.int().to(device)
        features = g.ndata["feat"]
        
        if config.dataset == 'yelp':
            labels = g.ndata["label"].float()
        else:
            labels = g.ndata["label"]
            
        masks = g.ndata["train_mask"].bool(), g.ndata["val_mask"].bool(), g.ndata["test_mask"].bool()
        
    elif "proteins" not in config.dataset:
        data = DglNodePropPredDataset(name=config.dataset, root=config.data_path)
        split_idx = data.get_idx_split()
        
        g, labels = data[0]
        labels = torch.squeeze(labels, dim=1)
        g = g.int().to(device)
        features = g.ndata["feat"]
        labels = labels.to(device)
        
        train_mask = split_idx["train"]
        valid_mask = split_idx["valid"]
        test_mask = split_idx["test"]
        total_nodes = train_mask.shape[0] + valid_mask.shape[0] + test_mask.shape[0]
        
        train_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask_bin[train_mask] = 1
        valid_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        valid_mask_bin[valid_mask] = 1
        test_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask_bin[test_mask] = 1
        
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
        
    else:  # ogbn-proteins
        data, g, labels, train_idx, val_idx, test_idx = load_proteins(config.data_path)
        g = g.int().to(device)
        features = g.ndata["feat"]
        labels = labels.float().to(device)
        
        total_nodes = train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0]
        train_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask_bin[train_idx] = 1
        valid_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        valid_mask_bin[val_idx] = 1
        test_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask_bin[test_idx] = 1
        
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
    
    # Prepare graph for custom kernels
    g = prepare_graph_for_kernels(g)
    
    # Add self loops if needed
    if config.selfloop:
        g = dgl.add_self_loop(g)
    
    # Model setup
    in_size = features.shape[1]
    out_size = data.num_classes
    if config.dataset == 'ogbn-proteins':
        out_size = 112
    
    logger.info(f"📊 Dataset: {config.dataset}")
    logger.info(f"📊 Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
    logger.info(f"📊 Features: {in_size}, Classes: {out_size}")
    logger.info(f"📊 MaxK value: {config.maxk}")
    
    # Create model with integrated MaxK kernels
    if config.model == 'sage':
        model = MaxKSAGE(
            in_size, config.hidden_dim, config.hidden_layers, out_size,
            config.maxk, feat_drop=config.dropout, norm=config.norm,
            nonlinear=config.nonlinear
        ).to(device)
    elif config.model == 'gcn':
        model = MaxKGCN(
            in_size, config.hidden_dim, config.hidden_layers, out_size,
            config.maxk, feat_drop=config.dropout, norm=config.norm,
            nonlinear=config.nonlinear
        ).to(device)
    elif config.model == 'gin':
        model = MaxKGIN(
            in_size, config.hidden_dim, config.hidden_layers, out_size,
            config.maxk, feat_drop=config.dropout, norm=config.norm,
            nonlinear=config.nonlinear
        ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Model parameters: {total_params:,}")
    
    # Training
    logger.info("🏃 Starting training...")
    best_val_acc, best_test_acc, avg_epoch_time = train(
        g, features, labels, masks, model, config, logger, writer
    )
    
    # Final evaluation
    logger.info("🧪 Final testing...")
    final_test_acc = evaluate(g, features, labels, masks[2], model, config, logger)
    logger.info(f"📊 Final test accuracy: {final_test_acc:.4f}")
    
    # Performance summary
    logger.info(f"\n🎯 PERFORMANCE SUMMARY")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Model: {config.model}")
    logger.info(f"MaxK: {config.maxk}")
    logger.info(f"Kernels: {'Custom' if KERNELS_AVAILABLE else 'DGL Fallback'}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best test accuracy: {best_test_acc:.4f}")
    logger.info(f"Average epoch time: {avg_epoch_time:.3f}s")
    
    # Save results
    results = {
        'dataset': config.dataset,
        'model': config.model,
        'maxk': config.maxk,
        'kernels_used': KERNELS_AVAILABLE,
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'avg_epoch_time': avg_epoch_time,
        'total_params': total_params
    }
    
    torch.save(results, os.path.join(config.path, 'results.pt'))
    logger.info(f"💾 Results saved to {config.path}/results.pt")

if __name__ == "__main__":
    main()