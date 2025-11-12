#!/usr/bin/env python3
"""
Unified script for generating AIGs using baseline models.

This script provides a command-line interface to generate AIGs using
Circuit Transformer, LayerDAG, or D-VAE baseline models.

Usage:
    python run_baseline.py --baseline circuit_transformer --num_samples 100
    python run_baseline.py --baseline layerdag --num_samples 100 --config path/to/config.yaml
    python run_baseline.py --baseline dvae --num_samples 100 --output results/
"""

import argparse
import yaml
import json
import os
from pathlib import Path
import sys
import torch
import numpy as np
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR.parent))

from baselines.circuit_transformer import CircuitTransformerBaseline
from baselines.layerdag import LayerDAGBaseline
from baselines.dvae import DVAEBaseline


BASELINE_MAP = {
    'circuit_transformer': CircuitTransformerBaseline,
    'layerdag': LayerDAGBaseline,
    'dvae': DVAEBaseline
}


def load_config(baseline_name: str, config_path: str = None) -> dict:
    """
    Load configuration for a baseline model.
    
    Args:
        baseline_name: Name of the baseline ('circuit_transformer', 'layerdag', 'dvae')
        config_path: Optional path to custom config file
        
    Returns:
        Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Load default config
        default_config_path = SCRIPT_DIR / baseline_name / 'config.yaml'
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            print(f"Warning: No config file found for {baseline_name}, using defaults")
            config = {'model': {}, 'generation': {}}
    
    return config


def save_generated_aigs(aigs: list, output_path: str, baseline_name: str):
    """
    Save generated AIGs to file.
    
    Args:
        aigs: List of generated AIGs
        output_path: Directory to save results
        baseline_name: Name of the baseline model
    """
    os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f"{baseline_name}_generated_{timestamp}.json")
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(aigs, f, indent=2)
    
    print(f"Saved {len(aigs)} generated AIGs to {output_file}")
    
    # Also save statistics
    stats_file = os.path.join(output_path, f"{baseline_name}_stats_{timestamp}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Baseline: {baseline_name}\n")
        f.write(f"Number of AIGs: {len(aigs)}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        # Compute statistics
        num_nodes_list = [len(aig['nodes']) for aig in aigs]
        num_edges_list = [len(aig['edges']) for aig in aigs]
        num_and_list = [aig.get('num_and_gates', 0) for aig in aigs]
        
        f.write(f"Node statistics:\n")
        f.write(f"  Mean: {np.mean(num_nodes_list):.2f}\n")
        f.write(f"  Std: {np.std(num_nodes_list):.2f}\n")
        f.write(f"  Min: {np.min(num_nodes_list)}\n")
        f.write(f"  Max: {np.max(num_nodes_list)}\n\n")
        
        f.write(f"Edge statistics:\n")
        f.write(f"  Mean: {np.mean(num_edges_list):.2f}\n")
        f.write(f"  Std: {np.std(num_edges_list):.2f}\n")
        f.write(f"  Min: {np.min(num_edges_list)}\n")
        f.write(f"  Max: {np.max(num_edges_list)}\n\n")
        
        f.write(f"AND gate statistics:\n")
        f.write(f"  Mean: {np.mean(num_and_list):.2f}\n")
        f.write(f"  Std: {np.std(num_and_list):.2f}\n")
        f.write(f"  Min: {np.min(num_and_list)}\n")
        f.write(f"  Max: {np.max(num_and_list)}\n")
    
    print(f"Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate AIGs using baseline models')
    parser.add_argument('--baseline', type=str, required=True,
                       choices=['circuit_transformer', 'layerdag', 'dvae'],
                       help='Baseline model to use')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of AIGs to generate')
    parser.add_argument('--max_nodes', type=int, default=50,
                       help='Maximum number of AND nodes')
    parser.add_argument('--num_inputs', type=int, default=4,
                       help='Number of primary inputs')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./baseline_results',
                       help='Output directory for generated AIGs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"=" * 80)
    print(f"Baseline AIG Generation")
    print(f"=" * 80)
    print(f"Baseline: {args.baseline}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Max AND nodes: {args.max_nodes}")
    print(f"Number of inputs: {args.num_inputs}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"=" * 80)
    
    # Load configuration
    config = load_config(args.baseline, args.config)
    
    # Override config with command-line arguments
    if 'model' not in config:
        config['model'] = {}
    config['model']['max_nodes'] = args.max_nodes
    
    # Initialize baseline model
    baseline_class = BASELINE_MAP[args.baseline]
    model = baseline_class(config['model'])
    
    # Load pretrained weights if provided
    model.load_pretrained(args.checkpoint)
    
    print(f"\nGenerating {args.num_samples} AIGs...")
    
    # Generate AIGs
    generated_aigs = model.generate(
        num_samples=args.num_samples,
        max_nodes=args.max_nodes,
        temperature=args.temperature,
        num_inputs=args.num_inputs
    )
    
    print(f"Generated {len(generated_aigs)} AIGs")
    
    # Save results
    save_generated_aigs(generated_aigs, args.output, args.baseline)
    
    print(f"\n" + "=" * 80)
    print(f"Generation complete!")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
