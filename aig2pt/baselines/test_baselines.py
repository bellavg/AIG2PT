"""
Test script for baseline models.

This script tests that all baseline models can be initialized and generate AIGs.
"""

import sys
from pathlib import Path
import tempfile
import json

# Add baselines to path
BASELINES_DIR = Path(__file__).parent
sys.path.insert(0, str(BASELINES_DIR))

from circuit_transformer import CircuitTransformerBaseline
from layerdag import LayerDAGBaseline
from dvae import DVAEBaseline


def test_baseline_generation(baseline_class, baseline_name, config):
    """Test a single baseline model."""
    print(f"\nTesting {baseline_name}...")
    
    # Initialize model
    model = baseline_class(config)
    model.load_pretrained(None)
    
    # Generate a few AIGs
    num_samples = 3
    aigs = model.generate(
        num_samples=num_samples,
        max_nodes=10,
        temperature=1.0,
        num_inputs=4
    )
    
    # Verify output
    assert len(aigs) == num_samples, f"Expected {num_samples} AIGs, got {len(aigs)}"
    
    for i, aig in enumerate(aigs):
        # Check required keys
        assert 'nodes' in aig, f"AIG {i} missing 'nodes'"
        assert 'edges' in aig, f"AIG {i} missing 'edges'"
        assert 'edge_types' in aig, f"AIG {i} missing 'edge_types'"
        
        # Check structure
        assert len(aig['nodes']) > 0, f"AIG {i} has no nodes"
        assert len(aig['edges']) == len(aig['edge_types']), \
            f"AIG {i} edge count mismatch"
        
        # Verify node types are valid
        valid_node_types = {'CONST0', 'PI', 'AND'}
        for node in aig['nodes']:
            assert node in valid_node_types, f"Invalid node type: {node}"
        
        # Verify edge types are valid
        valid_edge_types = {'REG', 'INV'}
        for edge_type in aig['edge_types']:
            assert edge_type in valid_edge_types, f"Invalid edge type: {edge_type}"
    
    print(f"✓ {baseline_name} passed all tests")
    return True


def main():
    """Run tests for all baseline models."""
    print("=" * 80)
    print("Testing Baseline Models for AIG Generation")
    print("=" * 80)
    
    # Test Circuit Transformer
    ct_config = {
        'n_embd': 128,
        'n_layer': 2,
        'n_head': 2,
        'max_nodes': 20,
        'vocab_size': 72
    }
    test_baseline_generation(CircuitTransformerBaseline, "Circuit Transformer", ct_config)
    
    # Test LayerDAG
    ld_config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'max_nodes': 20,
        'diffusion_steps': 5
    }
    test_baseline_generation(LayerDAGBaseline, "LayerDAG", ld_config)
    
    # Test D-VAE
    dv_config = {
        'hidden_dim': 64,
        'latent_dim': 32,
        'max_nodes': 20,
        'num_node_types': 3
    }
    test_baseline_generation(DVAEBaseline, "D-VAE", dv_config)
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == '__main__':
    main()
