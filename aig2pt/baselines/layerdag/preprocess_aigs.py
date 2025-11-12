#!/usr/bin/env python
"""
Preprocessing script to convert raw .aig files to PyTorch Geometric format for LayerDAG.

This script:
1. Reads .aig files using aigverse
2. Uses aigverse utilities (to_edge_list, to_networkx) for efficient conversion
3. Converts to PyG Data objects with node features and edge connectivity
4. Saves as .pt files for LayerDAG training

Usage:
    python preprocess_aigs.py \
        --input_dir /path/to/raw/aig/files \
        --output_dir /path/to/output
"""
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data

# Check for aigverse
try:
    from aigverse import read_aiger_into_aig, read_ascii_aiger_into_aig, to_edge_list
    # Try to import to_networkx if available
    try:
        from aigverse import to_networkx
        HAS_NETWORKX_SUPPORT = True
    except ImportError:
        HAS_NETWORKX_SUPPORT = False
except ImportError:
    print("Error: 'aigverse' library not found.")
    print("Please install it: pip install aigverse")
    sys.exit(1)


def aig_to_pyg_with_networkx(aig, aig_file):
    """
    Convert AIG to PyG using aigverse's to_networkx utility.
    
    This leverages aigverse's built-in NetworkX conversion which handles
    topological ordering and synthetic PO nodes automatically.
    
    Args:
        aig: aigverse.Aig object
        aig_file: Path to source file (for error reporting)
        
    Returns:
        PyG Data object
    """
    import networkx as nx
    
    # Convert to NetworkX graph using aigverse utility
    G = to_networkx(aig)
    
    # Build node type mapping from NetworkX graph
    node_types = []
    node_id_to_idx = {}
    
    # Process nodes in topological order
    for idx, node_id in enumerate(nx.topological_sort(G)):
        node_data = G.nodes[node_id]
        node_type = node_data.get('type', 'AND')
        
        # Map node types: 0=CONST, 1=PI, 2=AND, 3=PO
        if node_type == 'CONST' or node_id == 0:
            node_types.append(0)
        elif node_type == 'PI':
            node_types.append(1)
        elif node_type == 'AND':
            node_types.append(2)
        elif node_type == 'PO':
            node_types.append(3)
        else:
            node_types.append(2)  # Default to AND
        
        node_id_to_idx[node_id] = idx
    
    # Extract edges from NetworkX graph
    edge_src = []
    edge_dst = []
    edge_types = []
    
    for src, dst, edge_data in G.edges(data=True):
        src_idx = node_id_to_idx[src]
        dst_idx = node_id_to_idx[dst]
        edge_src.append(src_idx)
        edge_dst.append(dst_idx)
        
        # Edge type: 0=FWD, 1=INV
        is_inverted = edge_data.get('inverted', False) or edge_data.get('inv', False)
        edge_types.append(1 if is_inverted else 0)
    
    # Create PyG Data object
    if len(edge_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
    
    x = torch.tensor(node_types, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def aig_to_pyg_direct(aig, aig_file):
    """
    Convert AIG to PyG directly from aigverse AIG object.
    
    This is used when aigverse's to_networkx is not available.
    
    Args:
        aig: aigverse.Aig object
        aig_file: Path to source file (for error reporting)
        
    Returns:
        PyG Data object
    """
    # Build node type mapping
    # Node types: 0=CONST, 1=PI, 2=AND
    node_types = []
    node_id_to_idx = {}
    
    # Add constant node (always index 0)
    node_types.append(0)  # CONST
    node_id_to_idx[0] = 0
    current_idx = 1
    
    # Add primary inputs
    for pi_id in aig.inputs:
        node_types.append(1)  # PI
        node_id_to_idx[pi_id] = current_idx
        current_idx += 1
    
    # Add AND gates (aigverse provides them in topological order)
    for and_id in aig.and_gates:
        node_types.append(2)  # AND
        node_id_to_idx[and_id] = current_idx
        current_idx += 1
    
    # Build edges using aigverse's to_edge_list utility
    edge_src = []
    edge_dst = []
    edge_types = []
    
    # Use to_edge_list if available, otherwise process and_gates directly
    try:
        edge_list = to_edge_list(aig)
        for src_id, dst_id, is_inverted in edge_list:
            if src_id in node_id_to_idx and dst_id in node_id_to_idx:
                edge_src.append(node_id_to_idx[src_id])
                edge_dst.append(node_id_to_idx[dst_id])
                edge_types.append(1 if is_inverted else 0)
    except:
        # Fallback: process AND gates directly
        for and_id, (input0, input1) in aig.and_gates.items():
            dst_idx = node_id_to_idx[and_id]
            
            # Input 0
            src0_id = abs(input0) >> 1  # Remove inversion bit
            if src0_id in node_id_to_idx:
                src0_idx = node_id_to_idx[src0_id]
                edge_src.append(src0_idx)
                edge_dst.append(dst_idx)
                edge_types.append(1 if (input0 & 1) else 0)
            
            # Input 1
            src1_id = abs(input1) >> 1
            if src1_id in node_id_to_idx:
                src1_idx = node_id_to_idx[src1_id]
                edge_src.append(src1_idx)
                edge_dst.append(dst_idx)
                edge_types.append(1 if (input1 & 1) else 0)
    
    # Create PyG Data object
    if len(edge_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
    
    x = torch.tensor(node_types, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def aig_to_pyg(aig_file):
    """
    Convert a single .aig file to PyTorch Geometric Data object.
    
    Uses aigverse utilities (to_networkx or to_edge_list) when available
    for more robust conversion.
    
    Args:
        aig_file: Path to .aig file
        
    Returns:
        PyG Data object with:
            - x: Node type features (0=CONST, 1=PI, 2=AND, 3=PO)
            - edge_index: Edge connectivity [2, num_edges]
            - edge_attr: Edge types (0=FWD, 1=INV)
    """
    # Read AIG file
    try:
        # Try binary format first
        if str(aig_file).endswith('.aig'):
            aig = read_aiger_into_aig(str(aig_file))
        else:  # ASCII format (.aag)
            aig = read_ascii_aiger_into_aig(str(aig_file))
    except Exception as e:
        print(f"Warning: Could not read {aig_file}: {e}")
        return None
    
    # Use aigverse's to_networkx if available for better conversion
    if HAS_NETWORKX_SUPPORT:
        try:
            return aig_to_pyg_with_networkx(aig, aig_file)
        except Exception as e:
            # Fallback to direct conversion if networkx fails
            print(f"Warning: NetworkX conversion failed for {aig_file}, using direct method: {e}")
            return aig_to_pyg_direct(aig, aig_file)
    else:
        return aig_to_pyg_direct(aig, aig_file)


def process_directory(input_dir, split_name):
    """
    Process all .aig/.aag files in a directory.
    
    Args:
        input_dir: Directory containing .aig or .aag files
        split_name: Name of the split (e.g., 'train', 'val', 'test')
        
    Returns:
        List of PyG Data objects
    """
    data_list = []
    
    # Find all .aig and .aag files
    aig_files = list(Path(input_dir).glob('*.aig')) + list(Path(input_dir).glob('*.aag'))
    
    if len(aig_files) == 0:
        print(f"Warning: No .aig or .aag files found in {input_dir}")
        return data_list
    
    print(f"Processing {len(aig_files)} files from {split_name}...")
    
    for aig_file in tqdm(aig_files):
        data = aig_to_pyg(aig_file)
        if data is not None:
            data_list.append(data)
    
    print(f"Successfully processed {len(data_list)}/{len(aig_files)} files")
    
    return data_list


def main():
    parser = argparse.ArgumentParser(
        description='Convert raw .aig files to PyG format for LayerDAG'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing train/, val/, test/ subdirectories with .aig files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory where raw/ subdirectory will be created with .pt files'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_input = input_dir / split
        
        if not split_input.exists():
            print(f"Warning: {split_input} not found, skipping {split} split")
            continue
        
        # Process files
        data_list = process_directory(split_input, split)
        
        if len(data_list) > 0:
            # Save as .pt file
            output_file = raw_dir / f'{split}.pt'
            torch.save(data_list, output_file)
            print(f"Saved {len(data_list)} graphs to {output_file}")
        else:
            print(f"No data to save for {split} split")
    
    print("\nPreprocessing complete!")
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print(f"  python train_layerdag.py --data_dir {output_dir} --config_file configs/aig.yaml")


if __name__ == '__main__':
    main()
