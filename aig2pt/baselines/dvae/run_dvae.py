#!/usr/bin/env python3
"""
Convenience script to run D-VAE baseline experiments.

Usage:
    python run_dvae.py train --data-dir /path/to/data
    python run_dvae.py sample --checkpoint /path/to/checkpoint.pt
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aig2pt.baselines.dvae.train import main as train_main
from aig2pt.baselines.dvae.sample import main as sample_main


def main():
    parser = argparse.ArgumentParser(description='Run D-VAE baseline experiments')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train D-VAE model')
    train_parser.add_argument('--data-dir', type=str, required=True,
                             help='Path to processed data directory')
    train_parser.add_argument('--output-dir', type=str, default='./dvae_output',
                             help='Output directory for checkpoints')
    train_parser.add_argument('--max-n', type=int, default=56,
                             help='Maximum number of nodes')
    train_parser.add_argument('--nvt', type=int, default=6,
                             help='Number of vertex types')
    train_parser.add_argument('--hidden-size', type=int, default=256,
                             help='Hidden size for GRU')
    train_parser.add_argument('--latent-size', type=int, default=56,
                             help='Latent space dimension')
    train_parser.add_argument('--bidirectional', action='store_true',
                             help='Use bidirectional encoding')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--beta', type=float, default=1.0,
                             help='Weight for KL divergence')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Generate samples from trained model')
    sample_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to model checkpoint')
    sample_parser.add_argument('--output', type=str, default='./dvae_samples.json',
                              help='Output file for generated samples')
    sample_parser.add_argument('--num-samples', type=int, default=100,
                              help='Number of samples to generate')
    sample_parser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size for generation')
    sample_parser.add_argument('--temperature', type=float, default=1.0,
                              help='Sampling temperature')
    sample_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Monkey-patch sys.argv for train_main
        sys.argv = ['train.py']
        for key, value in vars(args).items():
            if key != 'command' and value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f'--{key.replace("_", "-")}')
                else:
                    sys.argv.extend([f'--{key.replace("_", "-")}', str(value)])
        train_main()
    
    elif args.command == 'sample':
        # Monkey-patch sys.argv for sample_main
        sys.argv = ['sample.py']
        for key, value in vars(args).items():
            if key != 'command' and value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f'--{key.replace("_", "-")}')
                else:
                    sys.argv.extend([f'--{key.replace("_", "-")}', str(value)])
        sample_main()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
