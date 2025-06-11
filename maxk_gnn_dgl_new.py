import argparse

import dgl
import dgl.nn as dglnn
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from torch.autograd import Function
import math
import time
from collections import OrderedDict, deque

from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset, FlickrDataset, YelpDataset
from utils.config import TrainConfig
import os
import utils.general_utils as general_utils
# from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils.proteins_loader import load_proteins
from utils.models import SAGE, GCN, GIN, GNN_res


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

pid = os.getpid()
print("Current process ID:", pid)

# Cache strategy implementations
class CacheBase:
    """Base class for GPU feature cache strategies"""
    def __init__(self, features, cache_size_ratio=0.05, device='cuda'):
        """
        Initialize the cache
        
        Args:
            features: The original node features
            cache_size_ratio: Ratio of nodes to cache (between 0 and 1)
            device: Device to store the cache ('cuda' or 'cpu')
        """
        self.features = features
        self.num_nodes = features.shape[0]
        self.cache_size = int(self.num_nodes * cache_size_ratio)
        self.device = device
        self.hit_count = 0
        self.miss_count = 0
        
    def get_features(self, node_indices):
        """
        Get features for the specified nodes
        
        Args:
            node_indices: Indices of nodes to get features for
            
        Returns:
            Tensor of node features
        """
        raise NotImplementedError
    
    def cache_stats(self):
        """
        Return cache hit statistics
        
        Returns:
            Dict with cache statistics
        """
        total = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total if total > 0 else 0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'total_accesses': total,
            'hit_ratio': hit_ratio,
            'cache_size': self.cache_size
        }
    
    def reset_stats(self):
        """Reset hit/miss counters"""
        self.hit_count = 0
        self.miss_count = 0
    
    def __str__(self):
        return f"{self.__class__.__name__} - Size: {self.cache_size}, Hit ratio: {self.cache_stats()['hit_ratio']:.4f}"


class DirectCache(CacheBase):
    """
    Strategy 1: Direct fetching without caching
    Always fetches directly from the original tensor in small batches
    """
    def __init__(self, features, cache_size_ratio=0, device='cuda'):
        super().__init__(features, cache_size_ratio, device)
        # Store features on CPU
        if torch.cuda.is_available() and features.device.type == 'cuda':
            self.features = features.cpu()
        else:
            self.features = features
        # Maximum batch size for loading features
        self.batch_size = 10000
        
    def get_features(self, node_indices):
        """Directly fetch features from the original tensor in batches"""
        # Ensure indices are the correct type
        if not torch.is_tensor(node_indices):
            node_indices = torch.tensor(node_indices, dtype=torch.long, device='cpu')
        elif node_indices.device.type == 'cuda':
            node_indices = node_indices.cpu()
            
        node_indices = node_indices.long()
        
        # Get result size and create output tensor
        num_nodes = node_indices.size(0)
        result = torch.zeros((num_nodes, self.features.shape[1]), 
                             dtype=self.features.dtype, 
                             device=self.device)
        
        # Process in batches to avoid OOM
        for i in range(0, num_nodes, self.batch_size):
            end_idx = min(i + self.batch_size, num_nodes)
            batch_indices = node_indices[i:end_idx]
            result[i:end_idx] = self.features[batch_indices].to(self.device)
            
        self.miss_count += num_nodes
        return result


# class StaticOutDegreeCache(CacheBase):
#     """
#     Strategy 2: Static-OutD
#     Caches nodes with the largest out-degrees
#     """
#     def __init__(self, features, graph, cache_size_ratio=0.3, device='cuda'):
#         super().__init__(features, cache_size_ratio, device)
        
#         # Get out-degrees for all nodes
#         out_degrees = graph.out_degrees()
        
#         # Get indices of nodes with highest out-degrees
#         _, indices = torch.topk(out_degrees, self.cache_size)
#         self.cached_indices = set(indices.cpu().numpy())
        
#         # Create cache
#         self.cache = {idx.item(): features[idx].to(self.device) for idx in indices}
    
#     def get_features(self, node_indices):
#         """Get features, using cache for high out-degree nodes"""
#         result = torch.zeros((len(node_indices), self.features.shape[1]), 
#                              device=self.device, dtype=self.features.dtype)
        
#         cached_mask = torch.tensor([idx.item() in self.cached_indices for idx in node_indices], 
#                                   device=self.device)
#         non_cached_mask = ~cached_mask
        
#         # Process cached nodes
#         cached_indices = [i for i, idx in enumerate(node_indices) if idx.item() in self.cached_indices]
#         if cached_indices:
#             for i, idx in enumerate(node_indices):
#                 if idx.item() in self.cached_indices:
#                     result[i] = self.cache[idx.item()]
#             self.hit_count += len(cached_indices)
        
#         # Process non-cached nodes
#         non_cached_indices = [i for i, idx in enumerate(node_indices) if idx.item() not in self.cached_indices]
#         if non_cached_indices:
#             non_cached_nodes = node_indices[non_cached_mask]
#             non_cached_features = self.features[non_cached_nodes]
#             for i, idx in zip(non_cached_indices, non_cached_nodes):
#                 result[i] = non_cached_features[non_cached_nodes == idx][0]
#             self.miss_count += len(non_cached_indices)
        
#         return result


# class StaticPresamplingCache(CacheBase):
#     """
#     Strategy 3: Static-PreS
#     Caches nodes based on presampling results
#     """
#     def __init__(self, features, graph, num_samples=1000, fanout=10, cache_size_ratio=0.3, device='cuda'):
#         super().__init__(features, cache_size_ratio, device)
        
#         # Perform presampling to identify frequently accessed nodes
#         freq_dict = {}
        
#         for _ in range(num_samples):
#             # Randomly select seed nodes
#             seed_nodes = torch.randint(0, self.num_nodes, (100,), device=graph.device)
            
#             # Sample neighbors
#             for hop in range(2):  # 2-hop neighborhood sampling
#                 sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout])
#                 sampler_iter = dgl.dataloading.NodeDataLoader(
#                     graph, seed_nodes, sampler, batch_size=1, shuffle=False, drop_last=False
#                 )
                
#                 for _, _, blocks in sampler_iter:
#                     for block in blocks:
#                         nodes = block.srcdata[dgl.NID].cpu().numpy()
#                         for node in nodes:
#                             freq_dict[node] = freq_dict.get(node, 0) + 1
                            
#                 # Update seed nodes for next hop
#                 seed_nodes = torch.tensor(list(freq_dict.keys()), device=graph.device)
#                 if len(seed_nodes) > 1000:  # Limit number of seed nodes
#                     seed_nodes = seed_nodes[:1000]
        
#         # Get top nodes by access frequency
#         import heapq
#         cached_nodes = heapq.nlargest(self.cache_size, freq_dict.keys(), key=lambda x: freq_dict[x])
#         self.cached_indices = set(cached_nodes)
        
#         # Create cache
#         self.cache = {idx: features[idx].to(self.device) for idx in cached_nodes}
    
#     def get_features(self, node_indices):
#         """Get features, using cache for frequently sampled nodes"""
#         result = torch.zeros((len(node_indices), self.features.shape[1]), 
#                              device=self.device, dtype=self.features.dtype)
        
#         cached_mask = torch.tensor([idx.item() in self.cached_indices for idx in node_indices], 
#                                   device=self.device)
#         non_cached_mask = ~cached_mask
        
#         # Process cached nodes
#         cached_indices = [i for i, idx in enumerate(node_indices) if idx.item() in self.cached_indices]
#         if cached_indices:
#             for i, idx in enumerate(node_indices):
#                 if idx.item() in self.cached_indices:
#                     result[i] = self.cache[idx.item()]
#             self.hit_count += len(cached_indices)
        
#         # Process non-cached nodes
#         non_cached_indices = [i for i, idx in enumerate(node_indices) if idx.item() not in self.cached_indices]
#         if non_cached_indices:
#             non_cached_nodes = node_indices[non_cached_mask]
#             non_cached_features = self.features[non_cached_nodes]
#             for i, idx in zip(non_cached_indices, non_cached_nodes):
#                 result[i] = non_cached_features[non_cached_nodes == idx][0]
#             self.miss_count += len(non_cached_indices)
        
#         return result
class StaticOutDegreeCache(CacheBase):
    def __init__(self, features, graph, cache_size_ratio=0.3, device='cuda'):
        super().__init__(features, cache_size_ratio, device)
        
        # Store features on CPU
        self.features = features.cpu() if features.device.type == 'cuda' else features
            
        # Get out-degrees for all nodes
        out_degrees = graph.out_degrees()
        
        # Get indices of nodes with highest out-degrees
        _, indices = torch.topk(out_degrees, self.cache_size)
        self.cached_indices = set(indices.cpu().tolist())
        
        # Create cache
        self.cache = {}
        for idx in indices:
            idx_item = idx.item()
            self.cache[idx_item] = features[idx].to(self.device)
    
    def get_features(self, node_indices):
        """Simply return features for the requested node indices"""
        # Return a tensor of the same shape as node_indices but with features
        result = torch.zeros((node_indices.size(0), self.features.shape[1]), 
                        dtype=self.features.dtype, 
                        device=self.device)
        
        # Process each index
        hits = 0
        misses = 0
        
        indices = node_indices.cpu().tolist()
        # Process indices one by one to avoid unhashable type issues
        for i, idx in enumerate(indices):
            # Convert to a hashable type (int)
            if not isinstance(idx, int):
                try:
                    idx_item = int(idx)
                except:
                    continue
                
            if idx_item in self.cached_indices:
                result[i] = self.cache[idx_item]
                hits += 1
            else:
                result[i] = self.features[idx_item].to(self.device)
                misses += 1
        
        self.hit_count += hits
        self.miss_count += misses
        return result


class FIFOCache(CacheBase):
    """
    Strategy 4: FIFO
    Dynamic caching with First-In-First-Out policy
    """
    def __init__(self, features, cache_size_ratio=0.3, device='cuda'):
        super().__init__(features, cache_size_ratio, device)
        
        # Initialize empty cache and FIFO queue
        self.cache = {}
        self.queue = deque()
    
    def get_features(self, node_indices):
        """Get features with FIFO caching policy"""
        result = torch.zeros((len(node_indices), self.features.shape[1]), 
                             device=self.device, dtype=self.features.dtype)
        
        for i, idx in enumerate(node_indices):
            idx_item = idx.item()
            
            # Check if node is in cache
            if idx_item in self.cache:
                result[i] = self.cache[idx_item]
                self.hit_count += 1
            else:
                # Fetch feature from original tensor
                feature = self.features[idx].to(self.device)
                result[i] = feature
                self.miss_count += 1
                
                # Add to cache if not already full or remove oldest element
                if len(self.cache) >= self.cache_size:
                    # Remove oldest item
                    oldest = self.queue.popleft()
                    del self.cache[oldest]
                
                # Add new item to cache
                self.cache[idx_item] = feature
                self.queue.append(idx_item)
        
        return result


class LRUCache(CacheBase):
    """
    Strategy 5: LRU
    Dynamic caching with Least Recently Used policy
    """
    def __init__(self, features, cache_size_ratio=0.3, device='cuda'):
        super().__init__(features, cache_size_ratio, device)
        
        # Use OrderedDict for LRU tracking
        self.cache = OrderedDict()
    
    def get_features(self, node_indices):
        """Get features with LRU caching policy"""
        result = torch.zeros((len(node_indices), self.features.shape[1]), 
                             device=self.device, dtype=self.features.dtype)
        
        for i, idx in enumerate(node_indices):
            idx_item = idx.item()
            
            # Check if node is in cache
            if idx_item in self.cache:
                # Move to end (most recently used)
                feature = self.cache.pop(idx_item)
                self.cache[idx_item] = feature
                result[i] = feature
                self.hit_count += 1
            else:
                # Fetch feature from original tensor
                feature = self.features[idx].to(self.device)
                result[i] = feature
                self.miss_count += 1
                
                # Add to cache if not already full or remove LRU element
                if len(self.cache) >= self.cache_size:
                    # Remove least recently used item (first item)
                    self.cache.popitem(last=False)
                
                # Add new item to cache
                self.cache[idx_item] = feature
        
        return result
    
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
            # return general_utils.accuracy(logits[mask], labels[mask])[0]
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

def train(g, features, labels, masks, model, config, logger, writer):
    # Log initial feature dimensions
    logger.info(f"Input features shape: {features.shape}")
    
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    if (config.dataset != 'yelp') and (config.dataset != 'ogbn-proteins'):
        loss_fcn = nn.CrossEntropyLoss()
    else:
        loss_fcn = F.binary_cross_entropy_with_logits
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.w_lr, weight_decay=config.w_weight_decay)
    if config.enable_lookahead:
        optimizer = general_utils.Lookahead(optimizer)
    
    # training loop
    best_val_accuracy = 0
    best_test_accuracy = 0
    model.start_timing()
    
    for epoch in range(config.epochs):
        model.train()
        
        # Add hooks to track intermediate feature sizes
        intermediate_features = {}
        def hook_fn(name):
            def hook(module, input, output):
                intermediate_features[name] = output.shape
            return hook
        
        # Register hooks for each SAGE layer
        hooks = []
        for i, layer in enumerate(model.layers):
            hook = layer.register_forward_hook(hook_fn(f'sage_layer_{i}'))
            hooks.append(hook)
        
        logits = model(g, features)
        
        # Log feature sizes for first epoch
        if epoch == 0:
            logger.info("Feature dimensions through the model:")
            logger.info(f"After input linear layer: {model.lin_in(features).shape}")
            for name, shape in intermediate_features.items():
                logger.info(f"{name}: {shape}")
            logger.info(f"Final output shape: {logits.shape}")
        
        # Remove hooks after logging
        for hook in hooks:
            hook.remove()
        
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()
        
        train_acc, val_acc, test_acc = evaluate_masks(g, features, labels, masks, model, config, logger)
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_test_accuracy = test_acc
            
        writer.add_scalar('train/loss', loss.item(), epoch)
        writer.add_scalar('train/train_acc', train_acc, epoch)
        writer.add_scalar('train/val_acc', val_acc, epoch)
        writer.add_scalar('train/test_acc', test_acc, epoch)
        
        logger.info(
            "Epoch {:04d}/{:04d}| Loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f} | Test Accuracy {:.4f} | Best val. Accuracy {:.4f} | Best test Accuracy {:.4f}".format(
                epoch, config.epochs, loss.item(), train_acc, val_acc, test_acc, best_val_accuracy, best_test_accuracy
            )
        )
    
    stats = model.stop_timing()
    logger.info(f"Total training time: {stats['total_training_time']:.4f}s")
    logger.info(f"Forward aggregation time: {stats['aggregation_time']:.4f}s")
    logger.info(f"Aggregation percentage: {stats['percentage']:.2f}%")
    
# def train(g, features, labels, masks, model, config, logger, writer):
#     # define train/val samples, loss function and optimizer
#     train_mask = masks[0]
#     if (config.dataset != 'yelp') and (config.dataset != 'ogbn-proteins'):
#         loss_fcn = nn.CrossEntropyLoss()
#     else:
#         loss_fcn = F.binary_cross_entropy_with_logits
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.w_lr, weight_decay=config.w_weight_decay)
#     if config.enable_lookahead: 
#         optimizer = general_utils.Lookahead(optimizer)
#     # training loop
#     best_val_accuracy = 0
#     best_test_accuracy = 0
#     #logger.info(f"count start")
#     model.start_timing()
    
#     for epoch in range(config.epochs):
#         # torch.cuda.empty_cache()
#         model.train()
#         logits = model(g, features)
#         loss = loss_fcn(logits[train_mask], labels[train_mask])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # torch.cuda.empty_cache()

#         train_acc, val_acc, test_acc = evaluate_masks(g, features, labels, masks, model, config, logger)
#         if val_acc > best_val_accuracy:
#             best_val_accuracy = val_acc
#             best_test_accuracy = test_acc
#         writer.add_scalar('train/loss', loss.item(), epoch)
#         writer.add_scalar('train/train_acc', train_acc, epoch)
#         writer.add_scalar('train/val_acc', val_acc, epoch)
#         writer.add_scalar('train/test_acc', test_acc, epoch)
#         logger.info(
#             "Epoch {:04d}/{:04d}| Loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f} | Test Accuracy {:.4f} | Best val. Accuracy {:.4f} | Best test Accuracy {:.4f}".format(
#                 epoch, config.epochs, loss.item(), train_acc, val_acc, test_acc, best_val_accuracy, best_test_accuracy
#             )
#         )
#     stats = model.stop_timing()
#     logger.info(f"Total training time: {stats['total_training_time']:.4f}s")
#     logger.info(f"Forward aggregation time: {stats['aggregation_time']:.4f}s")
#     logger.info(f"Aggregation percentage: {stats['percentage']:.2f}%")

if __name__ == "__main__":

    config = TrainConfig()
  
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)

    logger = general_utils.get_logger(os.path.join(config.path, "{}.log".format(config.dataset)))
    config.print_params(logger.info)

    torch.cuda.set_device(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training with DGL built-in GraphConv module.")

    
    if "ogb" not in config.dataset:
        # load and preprocess dataset
        transform = (
            AddSelfLoop()
        )  # by default, it will first remove self-loops to prevent duplication
        if config.dataset == 'reddit':
            data = RedditDataset(transform=transform, raw_dir= config.data_path)
        elif config.dataset == 'flickr':
            data = FlickrDataset(transform=transform, raw_dir= config.data_path)
        elif config.dataset == 'yelp':
            data = YelpDataset(transform=transform, raw_dir= config.data_path)
        g = data[0]
        g = g.int().to(device)
        features = g.ndata["feat"]
        if config.dataset == 'yelp':
            labels = g.ndata["label"].float()#.float()
        else:
            labels = g.ndata["label"]
        masks = g.ndata["train_mask"].bool(), g.ndata["val_mask"].bool(), g.ndata["test_mask"].bool()
    elif "proteins" not in config.dataset:
        data = DglNodePropPredDataset(name=config.dataset, root = config.data_path)
        split_idx = data.get_idx_split()

        # there is only one graph in Node Property Prediction datasets
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
        # masks = split_idx["train"].bool().to(device), split_idx["valid"].bool().to(device), split_idx["test"].bool().to(device)
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
        # print(split_idx["train"])
        # print(split_idx["train"].shape)
        # print(masks[1])
        # print(masks[1].shape)
        # print(masks[2])
        # print(masks[2].shape)
    ##### ogbn_proteins loader
    else:
        data, g, labels, train_idx, val_idx, test_idx = load_proteins(config.data_path)
        g = g.int().to(device)
        features = g.ndata["feat"]
        labels = labels.float().to(device)
        ### Get the train, validation, and test mask
        total_nodes = train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0]
        train_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        train_mask_bin[train_idx] = 1
        valid_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        valid_mask_bin[val_idx] = 1
        test_mask_bin = torch.zeros(total_nodes, dtype=torch.bool)
        test_mask_bin[test_idx] = 1
        masks = train_mask_bin.to(device), valid_mask_bin.to(device), test_mask_bin.to(device)
        
    in_size = features.shape[1]
    out_size = data.num_classes
    if config.dataset == 'ogbn-proteins':
        out_size = 112
    if config.selfloop:
        g = dgl.add_self_loop(g)

    # Setup cache strategy based on configuration
    cache_strategy = None
    if config.cache_strategy != 'none':
        logger.info(f"Using cache strategy: {config.cache_strategy} with ratio: {config.cache_size_ratio}")
        
    if config.cache_strategy == 'direct':
        cache_strategy = DirectCache(features, device=device)
    elif config.cache_strategy == 'static-outd':
        cache_strategy = StaticOutDegreeCache(features, g, cache_size_ratio=config.cache_size_ratio, device=device)
    elif config.cache_strategy == 'static-pres':
        cache_strategy = StaticPresamplingCache(features, g, cache_size_ratio=config.cache_size_ratio, device=device)
    elif config.cache_strategy == 'fifo':
        cache_strategy = FIFOCache(features, cache_size_ratio=config.cache_size_ratio, device=device)
    elif config.cache_strategy == 'lru':
        cache_strategy = LRUCache(features, cache_size_ratio=config.cache_size_ratio, device=device)
     
    if config.model == 'sage':
        model = SAGE(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear,cache_strategy=cache_strategy).to(device)
    elif config.model == 'gcn':
        model = GCN(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)
    elif config.model == 'gin':
        model = GIN(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)
    elif config.model == 'gnn_res':
        model = GNN_res(in_size, config.hidden_dim, config.hidden_layers, out_size, config.maxk, feat_drop=config.dropout, norm=config.norm, nonlinear = config.nonlinear).to(device)


        # if config.dataset == 'ogbn-products':

    # model training
    logger.info("Training...")
    train(g, features, labels, masks, model, config, logger, writer)

    # test the model
    logger.info("Testing...")
    acc = evaluate(g, features, labels, masks[2], model, config, logger)
    logger.info("Test accuracy {:.4f}".format(acc))