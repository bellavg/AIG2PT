"""
Data loader for D-VAE baseline.

Converts AIG data from PyTorch Geometric format to the sequential format
expected by D-VAE.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from torch_geometric.data import Data
from collections import defaultdict


class AIGDataset(Dataset):
    """
    Dataset for loading AIG graphs in D-VAE format.
    
    Converts PyG Data objects (with node features and edge indices) into
    sequential representations suitable for D-VAE training.
    
    The sequential representation encodes each node as:
    [one_hot(node_type), binary_vector(connections_from_predecessors)]
    
    Nodes are ordered topologically.
    """
    
    def __init__(self, data_path, max_n, nvt, START_TYPE=0, END_TYPE=1, split='train'):
        """
        Args:
            data_path: Path to the processed data directory
            max_n: Maximum number of nodes
            nvt: Number of vertex types (including START and END)
            START_TYPE: Index for start token
            END_TYPE: Index for end token
            split: 'train', 'val', or 'test'
        """
        self.data_path = Path(data_path)
        self.max_n = max_n
        self.nvt = nvt
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.split = split
        
        # Load the data
        self.graphs = self._load_data()
        
        print(f"Loaded {len(self.graphs)} graphs for {split} split")
    
    def _load_data(self):
        """Load AIG data from PyG format."""
        # Try to load from processed PyG files
        pyg_file = self.data_path / 'raw' / f'{self.split}.pt'
        
        if not pyg_file.exists():
            raise FileNotFoundError(f"Data file not found: {pyg_file}")
        
        # Load PyG data
        print(f"Loading PyG data from {pyg_file}")
        data_list = torch.load(pyg_file, weights_only=False)
        
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        print(f"Loaded {len(data_list)} PyG graphs")
        
        # Convert each PyG graph to D-VAE sequential format
        converted_graphs = []
        for data in data_list:
            try:
                seq_graph = self._convert_pyg_to_sequential(data)
                if seq_graph is not None:
                    converted_graphs.append(seq_graph)
            except Exception as e:
                print(f"Warning: Failed to convert graph: {e}")
                continue
        
        return converted_graphs
    
    def _convert_pyg_to_sequential(self, data):
        """
        Convert PyG Data object to sequential representation.
        
        Args:
            data: PyG Data object with x (node features) and edge_index
        
        Returns:
            Tensor of shape [max_n-1, nvt + max_n-1] representing the sequential graph
        """
        num_nodes = data.x.size(0)
        
        if num_nodes > self.max_n:
            print(f"Warning: Graph with {num_nodes} nodes exceeds max_n={self.max_n}")
            return None
        
        # Get node types (assuming x contains one-hot or index encoding)
        if data.x.dim() == 1:
            node_types = data.x.long()
        else:
            # One-hot encoding
            node_types = data.x.argmax(dim=1).long()
        
        # Get edge information
        edge_index = data.edge_index
        
        # Build adjacency information: for each node, which predecessors it connects from
        adjacency = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Only consider edges from src to dst where src < dst (topological order)
            if src < dst:
                adjacency[dst].append(src)
        
        # Create sequential representation
        # Shape: [num_nodes, nvt + max_n - 1]
        seq_repr = torch.zeros(num_nodes, self.nvt + self.max_n - 1)
        
        for i in range(num_nodes):
            # One-hot encode node type
            node_type = node_types[i].item()
            
            # Skip if it's the END_TYPE (we'll add it at the end)
            if node_type == self.END_TYPE:
                continue
            
            # Ensure node type is within bounds
            if node_type >= self.nvt:
                # Map to a valid type (this shouldn't happen with proper data)
                node_type = self.START_TYPE
            
            seq_repr[i, node_type] = 1.0
            
            # Encode connections from predecessors
            for pred in adjacency.get(i, []):
                if pred < i and pred < self.max_n - 1:
                    seq_repr[i, self.nvt + pred] = 1.0
        
        # Pad to max_n - 1 (reserving one spot for potential END token)
        if num_nodes < self.max_n - 1:
            padding = torch.zeros(self.max_n - 1 - num_nodes, self.nvt + self.max_n - 1)
            # Mark padding with START_TYPE to distinguish from actual nodes
            padding[:, self.START_TYPE] = 1.0
            seq_repr = torch.cat([seq_repr, padding], dim=0)
        else:
            # Truncate if necessary
            seq_repr = seq_repr[:self.max_n - 1]
        
        return seq_repr
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """
        Returns:
            Sequential graph tensor of shape [max_n-1, nvt + max_n-1]
        """
        return self.graphs[idx]


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of graph tensors
    
    Returns:
        Batched tensor of shape [batch_size, max_n-1, nvt + max_n-1]
    """
    # Stack into batch
    batch_tensor = torch.stack(batch, dim=0)
    return batch_tensor


def load_aig_data(data_dir, max_n, nvt, batch_size=32, num_workers=0):
    """
    Load AIG datasets for train/val/test splits.
    
    Args:
        data_dir: Path to data directory
        max_n: Maximum number of nodes
        nvt: Number of vertex types
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader (or None if split doesn't exist)
    """
    from torch.utils.data import DataLoader
    
    START_TYPE = 0
    END_TYPE = 1
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = AIGDataset(
                data_path=data_dir,
                max_n=max_n,
                nvt=nvt,
                START_TYPE=START_TYPE,
                END_TYPE=END_TYPE,
                split=split
            )
            
            if len(dataset) > 0:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=True
                )
                loaders[split] = loader
            else:
                loaders[split] = None
                
        except FileNotFoundError:
            print(f"Warning: No data found for {split} split")
            loaders[split] = None
    
    return loaders.get('train'), loaders.get('val'), loaders.get('test')
