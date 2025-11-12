#!/usr/bin/env python3
"""
Example workflow for D-VAE baseline.

This script demonstrates a complete workflow:
1. Load or create sample data
2. Train D-VAE model
3. Generate samples
4. Evaluate generated graphs

Usage:
    python example_dvae_workflow.py --data-dir /path/to/data --output-dir ./dvae_experiment
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure aig2pt is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_sample_data(output_dir):
    """
    Create sample data for testing (when real data is not available).
    
    In production, you would use actual AIG data from the dataset.
    """
    import torch
    from torch_geometric.data import Data
    
    print("Creating sample data...")
    
    # Create a few sample graphs
    graphs = []
    for i in range(50):  # Create 50 sample graphs
        num_nodes = torch.randint(5, 20, (1,)).item()
        
        # Random node features (one-hot encoded types)
        # Types: 0=START, 1=END, 2=CONST0, 3=PI, 4=AND, 5=PO
        x = torch.zeros(num_nodes, 6)
        for j in range(num_nodes):
            node_type = torch.randint(2, 5, (1,)).item()  # Random type (2-4)
            x[j, node_type] = 1.0
        
        # Random edges (ensuring DAG structure: src < dst)
        edge_list = []
        for src in range(num_nodes - 1):
            # Each node connects to 1-2 future nodes
            num_edges = torch.randint(0, 3, (1,)).item()
            for _ in range(num_edges):
                dst = torch.randint(src + 1, num_nodes, (1,)).item()
                edge_list.append([src, dst])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        graphs.append(Data(x=x, edge_index=edge_index))
    
    # Save as train/val/test splits
    splits = {
        'train': graphs[:40],
        'val': graphs[40:45],
        'test': graphs[45:]
    }
    
    raw_dir = Path(output_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_graphs in splits.items():
        save_path = raw_dir / f'{split_name}.pt'
        torch.save(split_graphs, save_path)
        print(f"  Saved {len(split_graphs)} graphs to {save_path}")
    
    print(f"✓ Sample data created in {output_dir}")


def train_model(data_dir, output_dir, epochs=20):
    """Train D-VAE model."""
    print(f"\nTraining D-VAE model...")
    
    from aig2pt.baselines.dvae.train import train_dvae
    import argparse
    
    # Create args for training
    class Args:
        pass
    
    args = Args()
    args.data_dir = str(data_dir)
    args.output_dir = str(output_dir / 'checkpoints')
    args.max_n = 56
    args.nvt = 6
    args.hidden_size = 128
    args.latent_size = 32
    args.bidirectional = False
    args.epochs = epochs
    args.batch_size = 8
    args.lr = 1e-3
    args.beta = 1.0
    args.eps_scale = 1.0
    args.patience = 5
    args.save_interval = 5
    args.seed = 42
    args.no_cuda = False
    args.num_workers = 0
    
    try:
        train_dvae(args)
        print("✓ Training completed")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_samples(checkpoint_path, output_path, num_samples=20):
    """Generate samples from trained model."""
    print(f"\nGenerating {num_samples} samples...")
    
    from aig2pt.baselines.dvae.sample import sample_dvae
    
    class Args:
        pass
    
    args = Args()
    args.checkpoint = str(checkpoint_path)
    args.output = str(output_path)
    args.num_samples = num_samples
    args.batch_size = 8
    args.temperature = 1.0
    args.use_prior = True
    args.seed = 42
    args.no_cuda = False
    
    try:
        sample_dvae(args)
        print("✓ Generation completed")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_samples(samples_path):
    """Analyze generated samples."""
    print(f"\nAnalyzing samples...")
    
    with open(samples_path, 'r') as f:
        samples = json.load(f)
    
    print(f"  Total samples: {len(samples)}")
    
    # Compute statistics
    num_nodes = [s['num_nodes'] for s in samples]
    num_edges = [len(s['edges']) for s in samples]
    
    import numpy as np
    
    print(f"\nStatistics:")
    print(f"  Nodes - Mean: {np.mean(num_nodes):.2f}, Std: {np.std(num_nodes):.2f}")
    print(f"  Edges - Mean: {np.mean(num_edges):.2f}, Std: {np.std(num_edges):.2f}")
    
    # Count node types
    node_types = {}
    for sample in samples:
        for node in sample['nodes']:
            t = node['type']
            node_types[t] = node_types.get(t, 0) + 1
    
    print(f"\nNode type distribution:")
    for t, count in sorted(node_types.items()):
        print(f"  {t}: {count}")
    
    print("✓ Analysis completed")


def main():
    parser = argparse.ArgumentParser(description='D-VAE baseline example workflow')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory (if None, creates sample data)')
    parser.add_argument('--output-dir', type=str, default='./dvae_experiment',
                       help='Output directory for experiment')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to generate')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing checkpoint)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("D-VAE Baseline Example Workflow")
    print("=" * 60)
    
    # Step 1: Prepare data
    if args.data_dir is None:
        print("\nNo data directory provided, creating sample data...")
        data_dir = output_dir / 'sample_data'
        create_sample_data(data_dir)
    else:
        data_dir = Path(args.data_dir)
        print(f"\nUsing data from: {data_dir}")
    
    # Step 2: Train model
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    
    if not args.skip_training:
        success = train_model(data_dir, output_dir, epochs=args.epochs)
        if not success:
            print("\n✗ Workflow failed at training step")
            return 1
    else:
        print(f"\nSkipping training, using checkpoint: {checkpoint_path}")
    
    # Step 3: Generate samples
    if not checkpoint_path.exists():
        print(f"\n✗ Checkpoint not found: {checkpoint_path}")
        return 1
    
    samples_path = output_dir / 'generated_samples.json'
    success = generate_samples(checkpoint_path, samples_path, args.num_samples)
    if not success:
        print("\n✗ Workflow failed at generation step")
        return 1
    
    # Step 4: Analyze samples
    analyze_samples(samples_path)
    
    print("\n" + "=" * 60)
    print("✓ Workflow completed successfully!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Samples: {samples_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
