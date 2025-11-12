"""
Sampling script for D-VAE baseline.

Generate new AIG graphs from trained D-VAE model.
"""

import torch
import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from .model import DVAE_AIG


def load_model(checkpoint_path, device):
    """Load trained D-VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load args if available
    model_dir = Path(checkpoint_path).parent
    args_path = model_dir / 'args.json'
    
    if args_path.exists():
        with open(args_path, 'r') as f:
            train_args = json.load(f)
        
        model = DVAE_AIG(
            max_n=train_args['max_n'],
            nvt=train_args['nvt'],
            START_TYPE=0,
            END_TYPE=1,
            hs=train_args['hidden_size'],
            nz=train_args['latent_size'],
            bidirectional=train_args.get('bidirectional', False)
        ).to(device)
    else:
        # Use default args
        print("Warning: args.json not found, using default model configuration")
        model = DVAE_AIG(
            max_n=56,
            nvt=6,
            START_TYPE=0,
            END_TYPE=1,
            hs=256,
            nz=56,
            bidirectional=False
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model


def graph_to_aig_format(graph_dict):
    """
    Convert D-VAE graph representation to AIG format.
    
    Args:
        graph_dict: Dict with 'node_types', 'edges', 'num_nodes'
    
    Returns:
        Dict with AIG structure information
    """
    node_types = graph_dict['node_types']
    edges = graph_dict['edges']
    num_nodes = graph_dict['num_nodes']
    
    # Create node type names (assuming: 0=START, 1=END, 2=CONST0, 3=PI, 4=AND, 5=PO)
    type_names = ['START', 'END', 'CONST0', 'PI', 'AND', 'PO']
    
    nodes = []
    for i, node_type in enumerate(node_types):
        if node_type < len(type_names):
            nodes.append({
                'id': i,
                'type': type_names[node_type],
                'type_id': node_type
            })
        else:
            nodes.append({
                'id': i,
                'type': 'UNKNOWN',
                'type_id': node_type
            })
    
    edge_list = [{'from': src, 'to': dst} for src, dst in edges]
    
    return {
        'num_nodes': num_nodes,
        'nodes': nodes,
        'edges': edge_list
    }


def sample_dvae(args):
    """Main sampling function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    
    all_graphs = []
    batch_size = args.batch_size
    num_batches = (args.num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc='Generating'):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, args.num_samples - batch_idx * batch_size)
            
            # Sample from latent space
            if args.use_prior:
                # Sample from prior N(0, I)
                z = torch.randn(current_batch_size, model.nz).to(device)
            else:
                # Sample from a learned distribution (if we had encoded real data)
                z = torch.randn(current_batch_size, model.nz).to(device) * args.temperature
            
            # Generate graphs
            graphs = model.generate(current_batch_size, z=z)
            
            # Convert to AIG format
            for graph in graphs:
                aig_graph = graph_to_aig_format(graph)
                all_graphs.append(aig_graph)
    
    print(f"Generated {len(all_graphs)} graphs")
    
    # Save samples
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_graphs, f, indent=2)
    
    print(f"Saved samples to {output_path}")
    
    # Print statistics
    print("\nGeneration statistics:")
    num_nodes_list = [g['num_nodes'] for g in all_graphs]
    num_edges_list = [len(g['edges']) for g in all_graphs]
    
    print(f"  Nodes - Mean: {np.mean(num_nodes_list):.2f}, "
          f"Std: {np.std(num_nodes_list):.2f}, "
          f"Min: {np.min(num_nodes_list)}, "
          f"Max: {np.max(num_nodes_list)}")
    
    print(f"  Edges - Mean: {np.mean(num_edges_list):.2f}, "
          f"Std: {np.std(num_edges_list):.2f}, "
          f"Min: {np.min(num_edges_list)}, "
          f"Max: {np.max(num_edges_list)}")
    
    # Count node types
    node_type_counts = {}
    for g in all_graphs:
        for node in g['nodes']:
            node_type = node['type']
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    
    print("\nNode type distribution:")
    for node_type, count in sorted(node_type_counts.items()):
        print(f"  {node_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Sample from trained D-VAE model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./dvae_samples.json',
                        help='Output file for generated samples')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (std dev for latent sampling)')
    parser.add_argument('--use-prior', action='store_true', default=True,
                        help='Sample from prior N(0,I) instead of learned distribution')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    sample_dvae(args)


if __name__ == '__main__':
    main()
