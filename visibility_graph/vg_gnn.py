"""
Visibility Graph GNN Module

Graph Neural Network architectures for Visibility Graph-based prediction.

This module requires PyTorch and PyTorch Geometric.
If not installed, features gracefully degrade to non-GNN methods.

Architecture:
1. Input: Visibility Graph with node features (price, volume, return)
2. GNN Layers: Graph Convolution or Graph Attention
3. Pooling: Global mean/max pool
4. Output: Classification (up/down) or Regression (return)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import warnings

# Try to import PyTorch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GNN features disabled.")

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    if TORCH_AVAILABLE:
        warnings.warn("PyTorch Geometric not available. Using fallback GNN.")


# =============================================================================
# GRAPH DATA PREPARATION
# =============================================================================

def prepare_node_features(
    series_window: np.ndarray,
    volume_window: Optional[np.ndarray] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Prepare node features for GNN from time series window.
    
    Each node (time step) has features:
    - Price (normalized)
    - Return (point-to-point)
    - Volume (if provided)
    - Time position (relative)
    
    Parameters
    ----------
    series_window : np.ndarray
        Price series of shape (window_size,)
    volume_window : np.ndarray, optional
        Volume series of shape (window_size,)
    normalize : bool
        Whether to normalize features
    
    Returns
    -------
    np.ndarray
        Node features of shape (window_size, num_features)
    """
    n = len(series_window)
    
    # Price feature (normalized)
    if normalize:
        price_norm = (series_window - series_window.mean()) / (series_window.std() + 1e-8)
    else:
        price_norm = series_window
    
    # Returns (1-step)
    returns = np.zeros(n)
    returns[1:] = (series_window[1:] - series_window[:-1]) / (series_window[:-1] + 1e-8)
    
    # Time position (normalized 0 to 1)
    time_pos = np.arange(n) / (n - 1)
    
    # Stack features
    features = [price_norm, returns, time_pos]
    
    # Add volume if available
    if volume_window is not None:
        if normalize:
            vol_norm = (volume_window - volume_window.mean()) / (volume_window.std() + 1e-8)
        else:
            vol_norm = volume_window
        features.append(vol_norm)
    
    return np.stack(features, axis=1).astype(np.float32)


def adjacency_to_edge_index_torch(adj: np.ndarray) -> 'torch.Tensor':
    """
    Convert adjacency matrix to PyTorch Geometric edge_index format.
    
    Parameters
    ----------
    adj : np.ndarray
        (n, n) adjacency matrix
    
    Returns
    -------
    torch.Tensor
        (2, num_edges) edge index
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    edges = np.array(np.where(adj > 0))
    return torch.from_numpy(edges).long()


def create_graph_data(
    adj: np.ndarray,
    node_features: np.ndarray,
    label: Optional[float] = None
) -> 'Data':
    """
    Create PyTorch Geometric Data object from adjacency and features.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    node_features : np.ndarray
        Node features (n, d)
    label : float, optional
        Target label
    
    Returns
    -------
    Data
        PyTorch Geometric Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("PyTorch Geometric not available")
    
    edge_index = adjacency_to_edge_index_torch(adj)
    x = torch.from_numpy(node_features).float()
    
    data = Data(x=x, edge_index=edge_index)
    
    if label is not None:
        data.y = torch.tensor([label]).float()
    
    return data


# =============================================================================
# GNN MODELS
# =============================================================================

if TORCH_AVAILABLE:
    
    class VisibilityGNN(nn.Module):
        """
        Graph Neural Network for Visibility Graph classification/regression.
        
        Architecture:
        - Multiple GCN/GAT layers
        - Global pooling
        - MLP head
        """
        
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 64,
            num_layers: int = 3,
            out_channels: int = 1,
            dropout: float = 0.2,
            use_attention: bool = False,
            task: str = 'classification'  # 'classification' or 'regression'
        ):
            super().__init__()
            
            self.task = task
            self.dropout = dropout
            
            # Graph convolution layers
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            
            # First layer
            if use_attention and TORCH_GEOMETRIC_AVAILABLE:
                self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=False))
            elif TORCH_GEOMETRIC_AVAILABLE:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                # Fallback: simple linear
                self.convs.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
            # Middle layers
            for _ in range(num_layers - 1):
                if use_attention and TORCH_GEOMETRIC_AVAILABLE:
                    self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
                elif TORCH_GEOMETRIC_AVAILABLE:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                else:
                    self.convs.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            
            # MLP head
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),  # *2 for concat of mean and max pool
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, out_channels)
            )
        
        def forward(self, x, edge_index, batch=None):
            """
            Forward pass.
            
            Parameters
            ----------
            x : Tensor
                Node features (num_nodes, in_channels)
            edge_index : Tensor
                Edge indices (2, num_edges)
            batch : Tensor, optional
                Batch assignment for each node
            
            Returns
            -------
            Tensor
                Predictions (batch_size, out_channels)
            """
            # Graph convolutions
            for conv, bn in zip(self.convs, self.bns):
                if TORCH_GEOMETRIC_AVAILABLE and isinstance(conv, (GCNConv, GATConv)):
                    x = conv(x, edge_index)
                else:
                    x = conv(x)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            if TORCH_GEOMETRIC_AVAILABLE:
                x_mean = global_mean_pool(x, batch)
                x_max = global_max_pool(x, batch)
            else:
                # Fallback: simple mean/max
                x_mean = x.mean(dim=0, keepdim=True)
                x_max = x.max(dim=0, keepdim=True)[0]
            
            x = torch.cat([x_mean, x_max], dim=1)
            
            # MLP head
            out = self.mlp(x)
            
            if self.task == 'classification':
                out = torch.sigmoid(out)
            
            return out
    
    
    class SimpleVisibilityMLP(nn.Module):
        """
        Fallback MLP model when PyTorch Geometric is not available.
        
        Uses flattened adjacency + node features as input.
        """
        
        def __init__(
            self,
            window_size: int,
            node_features: int = 3,
            hidden_dim: int = 128,
            task: str = 'classification'
        ):
            super().__init__()
            
            self.task = task
            
            # Input: flattened adj (upper triangle) + node features
            adj_size = window_size * (window_size - 1) // 2
            input_size = adj_size + window_size * node_features
            
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        def forward(self, adj_flat, node_features_flat):
            x = torch.cat([adj_flat, node_features_flat], dim=1)
            out = self.mlp(x)
            
            if self.task == 'classification':
                out = torch.sigmoid(out)
            
            return out


# =============================================================================
# DATASET
# =============================================================================

if TORCH_AVAILABLE:
    
    class VisibilityGraphDataset(Dataset):
        """
        PyTorch Dataset for Visibility Graph data.
        
        Handles lazy loading and caching of graphs.
        """
        
        def __init__(
            self,
            prices: np.ndarray,
            volumes: Optional[np.ndarray],
            labels: np.ndarray,
            window_size: int = 20,
            use_hvg: bool = True,
            precompute: bool = True
        ):
            """
            Parameters
            ----------
            prices : np.ndarray
                Price series (n_samples,)
            volumes : np.ndarray, optional
                Volume series (n_samples,)
            labels : np.ndarray
                Target labels (n_samples - window_size,)
            window_size : int
                Window size for VG construction
            use_hvg : bool
                Use HVG (faster) or VG
            precompute : bool
                Precompute all graphs (uses more memory but faster)
            """
            from .vg_core import construct_hvg_fast, construct_vg_fast
            
            self.prices = prices
            self.volumes = volumes
            self.labels = labels
            self.window_size = window_size
            self.use_hvg = use_hvg
            
            self.construct_fn = construct_hvg_fast if use_hvg else construct_vg_fast
            
            # Number of valid samples
            self.n_samples = len(prices) - window_size
            
            # Precompute if requested
            self.graphs = None
            if precompute:
                self._precompute_graphs()
        
        def _precompute_graphs(self):
            """Precompute all graphs."""
            print(f"Precomputing {self.n_samples} graphs...")
            self.graphs = []
            
            for idx in range(self.n_samples):
                start = idx
                end = idx + self.window_size
                
                price_window = self.prices[start:end]
                vol_window = self.volumes[start:end] if self.volumes is not None else None
                
                adj = self.construct_fn(price_window)
                node_features = prepare_node_features(price_window, vol_window)
                
                if TORCH_GEOMETRIC_AVAILABLE:
                    label = self.labels[idx] if idx < len(self.labels) else 0.0
                    data = create_graph_data(adj, node_features, label)
                    self.graphs.append(data)
                else:
                    self.graphs.append((adj, node_features))
            
            print("Precomputation complete.")
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            if self.graphs is not None:
                return self.graphs[idx]
            
            # Lazy computation
            start = idx
            end = idx + self.window_size
            
            price_window = self.prices[start:end]
            vol_window = self.volumes[start:end] if self.volumes is not None else None
            
            adj = self.construct_fn(price_window)
            node_features = prepare_node_features(price_window, vol_window)
            
            if TORCH_GEOMETRIC_AVAILABLE:
                label = self.labels[idx] if idx < len(self.labels) else 0.0
                return create_graph_data(adj, node_features, label)
            else:
                return adj, node_features


# =============================================================================
# TRAINING
# =============================================================================

if TORCH_AVAILABLE:
    
    def train_visibility_gnn(
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        task: str = 'classification',
        device: str = 'auto',
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train Visibility GNN model.
        
        Parameters
        ----------
        model : nn.Module
            GNN model
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset, optional
            Validation dataset
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        weight_decay : float
            L2 regularization
        task : str
            'classification' or 'regression'
        device : str
            'auto', 'cuda', or 'cpu'
        verbose : bool
            Print progress
        
        Returns
        -------
        Tuple[nn.Module, Dict]
            Trained model and training history
        """
        # Device selection
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        model = model.to(device)
        
        # Data loaders
        if TORCH_GEOMETRIC_AVAILABLE:
            train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        if task == 'classification':
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for batch in train_loader:
                if TORCH_GEOMETRIC_AVAILABLE:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    target = batch.y.view(-1, 1)
                else:
                    adj, features = batch
                    adj = adj.to(device)
                    features = features.to(device)
                    out = model(adj, features)
                    target = batch[-1].to(device).view(-1, 1)
                
                loss = criterion(out, target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * len(target)
                total_samples += len(target)
                
                if task == 'classification':
                    preds = (out > 0.5).float()
                    total_correct += (preds == target).sum().item()
            
            avg_train_loss = total_loss / total_samples
            train_metric = total_correct / total_samples if task == 'classification' else 0
            
            history['train_loss'].append(avg_train_loss)
            history['train_metric'].append(train_metric)
            
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_samples = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        if TORCH_GEOMETRIC_AVAILABLE:
                            batch = batch.to(device)
                            out = model(batch.x, batch.edge_index, batch.batch)
                            target = batch.y.view(-1, 1)
                        else:
                            adj, features = batch
                            out = model(adj.to(device), features.to(device))
                            target = batch[-1].to(device).view(-1, 1)
                        
                        val_loss += criterion(out, target).item() * len(target)
                        val_samples += len(target)
                        
                        if task == 'classification':
                            preds = (out > 0.5).float()
                            val_correct += (preds == target).sum().item()
                
                avg_val_loss = val_loss / val_samples
                val_metric = val_correct / val_samples if task == 'classification' else 0
                
                history['val_loss'].append(avg_val_loss)
                history['val_metric'].append(val_metric)
                
                scheduler.step(avg_val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}"
                if task == 'classification':
                    msg += f" | Train Acc: {train_metric:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {avg_val_loss:.4f}"
                    if task == 'classification':
                        msg += f" | Val Acc: {val_metric:.4f}"
                print(msg)
        
        return model, history


# =============================================================================
# INFERENCE
# =============================================================================

if TORCH_AVAILABLE:
    
    def predict_with_gnn(
        model: nn.Module,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        window_size: int = 20,
        use_hvg: bool = True,
        device: str = 'auto'
    ) -> np.ndarray:
        """
        Generate predictions using trained GNN.
        
        Parameters
        ----------
        model : nn.Module
            Trained GNN model
        prices : np.ndarray
            Price series
        volumes : np.ndarray, optional
            Volume series
        window_size : int
            Window size
        use_hvg : bool
            Use HVG or VG
        device : str
            Device to use
        
        Returns
        -------
        np.ndarray
            Predictions aligned with input (first window_size-1 values are NaN)
        """
        from .vg_core import construct_hvg_fast, construct_vg_fast
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        model = model.to(device)
        model.eval()
        
        construct_fn = construct_hvg_fast if use_hvg else construct_vg_fast
        
        n = len(prices)
        predictions = np.full(n, np.nan)
        
        with torch.no_grad():
            for idx in range(n - window_size):
                start = idx
                end = idx + window_size
                
                price_window = prices[start:end]
                vol_window = volumes[start:end] if volumes is not None else None
                
                adj = construct_fn(price_window)
                node_features = prepare_node_features(price_window, vol_window)
                
                if TORCH_GEOMETRIC_AVAILABLE:
                    data = create_graph_data(adj, node_features)
                    data = data.to(device)
                    out = model(data.x, data.edge_index)
                else:
                    # Fallback
                    adj_flat = torch.from_numpy(adj[np.triu_indices(window_size, k=1)]).float().unsqueeze(0)
                    feat_flat = torch.from_numpy(node_features.flatten()).float().unsqueeze(0)
                    out = model(adj_flat.to(device), feat_flat.to(device))
                
                predictions[end - 1] = out.item()
        
        return predictions


# =============================================================================
# UTILITIES
# =============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    return {
        'torch': TORCH_AVAILABLE,
        'torch_geometric': TORCH_GEOMETRIC_AVAILABLE,
        'cuda': TORCH_AVAILABLE and torch.cuda.is_available()
    }
