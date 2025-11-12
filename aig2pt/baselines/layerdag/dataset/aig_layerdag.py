"""
AIG Dataset adapter for LayerDAG.

Converts AIG data from raw .aig files (via PyG intermediate format) to LayerDAG's 
layer-based format for DAG generation.

Pipeline:
  1. Raw .aig files → preprocess_aigs.py → PyG Data objects (.pt files)
  2. PyG Data objects → AIGDAGDataset → LayerDAG format

AIGs (AND-Inverter Graphs) are represented as DAGs with:
- Node types: CONST (0), PI (Primary Input), AND gates
- Edge types: FWD (forward/non-inverting), INV (inverting)
"""
import os
import torch
from torch.utils.data import Dataset
from .general import DAGDataset


class AIGDAGDataset(DAGDataset):
    """
    AIG dataset in LayerDAG format.
    
    Extends the general DAGDataset to handle AIG-specific properties:
    - Maps AIG node types (CONST, PI, AND) to integer categories
    - Handles edge inversions as edge attributes
    - Supports unconditional generation (no labels)
    """
    
    def __init__(self, num_categories=3, label=False):
        """
        Initialize AIG DAG dataset.
        
        Args:
            num_categories: Number of node categories (3 for AIG: CONST, PI, AND)
            label: Whether to include labels for conditional generation
        """
        super().__init__(num_categories=num_categories, label=label)
        
        # AIG node type mapping
        self.node_type_map = {
            'CONST': 0,
            'PI': 1,
            'AND': 2
        }
        
    def add_aig_data(self, pyg_data, y=None):
        """
        Add an AIG from PyTorch Geometric format.
        
        Args:
            pyg_data: PyG Data object with x (node features), edge_index, edge_attr
            y: Optional label for conditional generation
        """
        # Extract node features (assume one-hot encoded or categorical)
        x_n = pyg_data.x
        if x_n.dim() > 1:
            # If one-hot encoded, convert to categorical
            x_n = x_n.argmax(dim=-1)
        
        # Extract edges (convert to src, dst format)
        edge_index = pyg_data.edge_index
        src = edge_index[0]  # Source nodes
        dst = edge_index[1]  # Destination nodes
        
        # Add to parent dataset
        self.add_data(src, dst, x_n, y)


def load_aig_from_pyg_files(data_dir, split='train'):
    """
    Load AIGs from PyTorch Geometric .pt files.
    
    Args:
        data_dir: Directory containing {split}.pt files
        split: One of 'train', 'val', 'test'
        
    Returns:
        AIGDAGDataset with loaded graphs
    """
    file_path = os.path.join(data_dir, 'raw', f'{split}.pt')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"AIG data file not found: {file_path}")
    
    print(f"Loading AIG data from {file_path}...")
    data_list = torch.load(file_path, weights_only=False)
    
    if not isinstance(data_list, list):
        raise ValueError(f"Expected list of Data objects, got {type(data_list)}")
    
    # Determine number of node categories from the first graph
    if len(data_list) > 0:
        sample_data = data_list[0]
        if hasattr(sample_data, 'x') and sample_data.x is not None:
            if sample_data.x.dim() > 1:
                num_categories = sample_data.x.shape[1]
            else:
                num_categories = sample_data.x.max().item() + 1
        else:
            num_categories = 3  # Default for AIG: CONST, PI, AND
    else:
        num_categories = 3
    
    # Create dataset
    dataset = AIGDAGDataset(num_categories=num_categories, label=False)
    
    # Add all graphs
    for data in data_list:
        dataset.add_aig_data(data)
    
    print(f"Loaded {len(dataset)} AIG graphs with {num_categories} node categories")
    
    return dataset


def load_aig_dataset(data_dir):
    """
    Load train, val, test splits of AIG dataset.
    
    Args:
        data_dir: Root directory containing raw/{train,val,test}.pt
        
    Returns:
        Tuple of (train_set, val_set, test_set)
    """
    splits = {}
    for split in ['train', 'val', 'test']:
        try:
            splits[split] = load_aig_from_pyg_files(data_dir, split)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {split} split: {e}")
            # Create empty dataset for missing splits
            splits[split] = AIGDAGDataset(num_categories=3, label=False)
    
    return splits['train'], splits['val'], splits['test']
