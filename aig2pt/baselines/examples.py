#!/usr/bin/env python3
"""
Example usage of baseline models for AIG generation.

This script demonstrates how to use the baseline models to generate AIGs
and compare them with the aig2pt model.
"""

import sys
from pathlib import Path
import json

# Add baselines to path
BASELINES_DIR = Path(__file__).parent
sys.path.insert(0, str(BASELINES_DIR))

from circuit_transformer import CircuitTransformerBaseline
from layerdag import LayerDAGBaseline
from dvae import DVAEBaseline


def print_aig_summary(aig, idx):
    """Print a summary of an AIG."""
    print(f"\nAIG #{idx}:")
    print(f"  Total nodes: {len(aig['nodes'])}")
    print(f"  Primary inputs: {aig.get('num_inputs', 'N/A')}")
    print(f"  AND gates: {aig.get('num_and_gates', 'N/A')}")
    print(f"  Total edges: {len(aig['edges'])}")
    
    # Count edge types
    inv_count = sum(1 for et in aig['edge_types'] if et == 'INV')
    reg_count = len(aig['edge_types']) - inv_count
    print(f"  Regular edges: {reg_count}")
    print(f"  Inverted edges: {inv_count}")
    
    # Show node types
    from collections import Counter
    node_counts = Counter(aig['nodes'])
    print(f"  Node breakdown: {dict(node_counts)}")


def example_circuit_transformer():
    """Example: Generate AIGs using Circuit Transformer."""
    print("=" * 80)
    print("Example 1: Circuit Transformer Baseline")
    print("=" * 80)
    
    # Configure model
    config = {
        'n_embd': 256,
        'n_layer': 4,
        'n_head': 4,
        'max_nodes': 30,
        'vocab_size': 72
    }
    
    # Initialize model
    model = CircuitTransformerBaseline(config)
    model.load_pretrained()
    
    # Generate AIGs
    print("\nGenerating 3 AIGs with Circuit Transformer...")
    aigs = model.generate(
        num_samples=3,
        max_nodes=20,
        temperature=0.9,
        num_inputs=5
    )
    
    # Display results
    for i, aig in enumerate(aigs, 1):
        print_aig_summary(aig, i)
    
    return aigs


def example_layerdag():
    """Example: Generate AIGs using LayerDAG."""
    print("\n" + "=" * 80)
    print("Example 2: LayerDAG Baseline")
    print("=" * 80)
    
    # Configure model
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'max_nodes': 30,
        'diffusion_steps': 10
    }
    
    # Initialize model
    model = LayerDAGBaseline(config)
    model.load_pretrained()
    
    # Generate AIGs
    print("\nGenerating 3 AIGs with LayerDAG...")
    aigs = model.generate(
        num_samples=3,
        max_nodes=15,
        temperature=1.0,
        num_inputs=4
    )
    
    # Display results
    for i, aig in enumerate(aigs, 1):
        print_aig_summary(aig, i)
    
    return aigs


def example_dvae():
    """Example: Generate AIGs using D-VAE."""
    print("\n" + "=" * 80)
    print("Example 3: D-VAE Baseline")
    print("=" * 80)
    
    # Configure model
    config = {
        'hidden_dim': 128,
        'latent_dim': 64,
        'max_nodes': 30,
        'num_node_types': 3
    }
    
    # Initialize model
    model = DVAEBaseline(config)
    model.load_pretrained()
    
    # Generate AIGs
    print("\nGenerating 3 AIGs with D-VAE...")
    aigs = model.generate(
        num_samples=3,
        max_nodes=25,
        temperature=1.2,  # Higher temperature for more diversity
        num_inputs=6
    )
    
    # Display results
    for i, aig in enumerate(aigs, 1):
        print_aig_summary(aig, i)
    
    return aigs


def example_comparison():
    """Example: Compare all three baselines."""
    print("\n" + "=" * 80)
    print("Example 4: Comparing All Baselines")
    print("=" * 80)
    
    baselines = {
        'Circuit Transformer': CircuitTransformerBaseline({
            'n_embd': 128, 'n_layer': 2, 'n_head': 2, 
            'max_nodes': 30, 'vocab_size': 72
        }),
        'LayerDAG': LayerDAGBaseline({
            'hidden_dim': 64, 'num_layers': 2, 
            'max_nodes': 30, 'diffusion_steps': 5
        }),
        'D-VAE': DVAEBaseline({
            'hidden_dim': 64, 'latent_dim': 32,
            'max_nodes': 30, 'num_node_types': 3
        })
    }
    
    # Generate with each baseline
    results = {}
    for name, model in baselines.items():
        print(f"\nGenerating with {name}...")
        model.load_pretrained()
        aigs = model.generate(num_samples=5, max_nodes=20, num_inputs=4)
        results[name] = aigs
        
        # Compute average stats
        avg_nodes = sum(len(aig['nodes']) for aig in aigs) / len(aigs)
        avg_edges = sum(len(aig['edges']) for aig in aigs) / len(aigs)
        avg_and = sum(aig.get('num_and_gates', 0) for aig in aigs) / len(aigs)
        
        print(f"  Average nodes: {avg_nodes:.1f}")
        print(f"  Average edges: {avg_edges:.1f}")
        print(f"  Average AND gates: {avg_and:.1f}")
    
    return results


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Baseline Models for AIG Generation - Examples")
    print("=" * 80)
    
    # Run examples
    example_circuit_transformer()
    example_layerdag()
    example_dvae()
    example_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use run_baseline.py to generate larger datasets")
    print("2. Compare with aig2pt model outputs")
    print("3. Evaluate using V.U.N. metrics (Validity, Uniqueness, Novelty)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
