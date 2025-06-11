import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.init as init
from .maxk_layers import MaxKSAGEConv, MaxKGCNConv, MaxKFunction

class MaxKSAGE(nn.Module):
    """
    GraphSAGE with integrated MaxK nonlinearity and custom kernels
    """
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32, 
                 feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        
        self.num_layers = num_hid_layers
        self.nonlinear = nonlinear
        
        # Input/output linear layers
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        
        # MaxK SAGE layers
        self.layers = nn.ModuleList()
        for i in range(num_hid_layers):
            if norm:
                norm_layer = nn.LayerNorm(hid_size, elementwise_affine=True)
            else:
                norm_layer = None
                
            self.layers.append(MaxKSAGEConv(
                hid_size, hid_size, 
                aggregator_type='mean',
                feat_drop=feat_drop,
                norm=norm_layer,
                maxk=maxk
            ))
        
        # MaxK function for each layer
        if nonlinear == "maxk":
            self.maxk_fn = MaxKFunction.apply
            self.k_values = [maxk] * num_hid_layers
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
    
    def forward(self, g, x):
        # Input transformation
        x = self.lin_in(x)
        
        # Process through MaxK SAGE layers
        for i, layer in enumerate(self.layers):
            if self.nonlinear == 'maxk':
                # MaxK is applied inside the custom SAGE layer
                x = layer(g, x)
            elif self.nonlinear == 'relu':
                x = F.relu(x)
                # Use standard aggregation (fallback mode)
                x = layer(g, x)
        
        # Output transformation
        x = self.lin_out(x)
        return x

class MaxKGCN(nn.Module):
    """
    GCN with integrated MaxK nonlinearity and custom kernels
    """
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32,
                 feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        
        self.num_layers = num_hid_layers
        self.nonlinear = nonlinear
        self.norm = norm
        
        # Dropout layers
        self.dropoutlayers = nn.ModuleList()
        # GCN layers using custom MaxK implementation
        self.gcnlayers = nn.ModuleList()
        # Normalization layers
        self.normlayers = nn.ModuleList()
        # Linear layers
        self.linlayers = nn.ModuleList()
        
        for i in range(num_hid_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            self.gcnlayers.append(MaxKGCNConv(hid_size, hid_size, maxk=maxk))
            
            if norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
            
            self.linlayers.append(Linear(hid_size, hid_size))
        
        # Input/output transformations
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        
        # MaxK function
        if nonlinear == "maxk":
            self.maxk_fn = MaxKFunction.apply
            self.k_values = [maxk] * num_hid_layers
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        for linear in self.linlayers:
            init.xavier_uniform_(linear.weight)
    
    def forward(self, g, x):
        # Input transformation with ReLU
        x = self.lin_in(x).relu()
        
        # Process through GCN layers
        for i in range(self.num_layers):
            # Linear transformation
            x = self.linlayers[i](x)
            
            # Apply nonlinearity before aggregation
            if self.nonlinear == 'maxk':
                # MaxK is applied inside the custom GCN layer
                pass  # MaxK will be applied in gcnlayers[i]
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            
            # Dropout
            x = self.dropoutlayers[i](x)
            
            # Graph convolution with MaxK
            x = self.gcnlayers[i](g, x)
            
            # Normalization
            if self.norm:
                x = self.normlayers[i](x)
        
        # Output transformation
        x = self.lin_out(x)
        return x

class MaxKGIN(nn.Module):
    """
    GIN with integrated MaxK nonlinearity and custom kernels
    """
    def __init__(self, in_size, hid_size, num_hid_layers, out_size, maxk=32,
                 feat_drop=0.5, norm=False, nonlinear="maxk"):
        super().__init__()
        
        self.num_layers = num_hid_layers
        self.nonlinear = nonlinear
        self.norm = norm
        
        # Layers
        self.dropoutlayers = nn.ModuleList()
        self.ginlayers = nn.ModuleList()
        self.normlayers = nn.ModuleList()
        self.linlayers = nn.ModuleList()
        
        for i in range(num_hid_layers):
            self.dropoutlayers.append(nn.Dropout(feat_drop))
            
            # Custom GIN layer with MaxK
            self.ginlayers.append(MaxKGINConv(hid_size, hid_size, maxk=maxk))
            
            if norm:
                self.normlayers.append(nn.LayerNorm(hid_size, elementwise_affine=True))
            
            self.linlayers.append(Linear(hid_size, hid_size))
        
        # Input/output transformations
        self.lin_in = Linear(in_size, hid_size)
        self.lin_out = Linear(hid_size, out_size)
        
        # MaxK function
        if nonlinear == "maxk":
            self.maxk_fn = MaxKFunction.apply
            self.k_values = [maxk] * num_hid_layers
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.lin_in.weight)
        init.xavier_uniform_(self.lin_out.weight)
        for linear in self.linlayers:
            init.xavier_uniform_(linear.weight)
    
    def forward(self, g, x):
        # Input transformation with ReLU
        x = self.lin_in(x).relu()
        
        # Process through GIN layers
        for i in range(self.num_layers):
            # Linear transformation
            x = self.linlayers[i](x)
            
            # Apply nonlinearity before aggregation
            if self.nonlinear == 'maxk':
                # MaxK is applied inside the custom GIN layer
                pass  # MaxK will be applied in ginlayers[i]
            elif self.nonlinear == 'relu':
                x = F.relu(x)
            
            # Dropout
            x = self.dropoutlayers[i](x)
            
            # Graph isomorphism convolution with MaxK
            x = self.ginlayers[i](g, x)
            
            # Normalization
            if self.norm:
                x = self.normlayers[i](x)
        
        # Output transformation
        x = self.lin_out(x)
        return x

class MaxKGINConv(nn.Module):
    """
    Custom GIN layer using MaxK nonlinearity and custom kernels
    """
    def __init__(self, in_feats, out_feats, learn_eps=True, maxk=32):
        super(MaxKGINConv, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.maxk = maxk
        
        # Learnable epsilon parameter
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([0]))
        else:
            self.register_buffer('eps', torch.FloatTensor([0]))
        
        # MLP for GIN
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        
        # MaxK function
        self.maxk_fn = MaxKFunction.apply
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            # Apply MaxK nonlinearity first
            feat_sparse = self.maxk_fn(feat, self.maxk)
            
            graph.ndata['h'] = feat_sparse
            
            # Aggregate neighbors
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            
            # Add self connection
            output = (1 + self.eps) * feat + graph.ndata['neigh']
            
            # Apply MLP
            output = self.mlp(output)
            
            return output