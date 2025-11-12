"""
Quick test for D-VAE baseline implementation.
"""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aig2pt.baselines.dvae.model import DVAE_AIG


def test_model_creation():
    """Test that model can be created."""
    print("Testing model creation...")
    model = DVAE_AIG(
        max_n=20,
        nvt=6,
        START_TYPE=0,
        END_TYPE=1,
        hs=128,
        nz=32,
        bidirectional=False
    )
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")
    
    model = DVAE_AIG(
        max_n=20,
        nvt=6,
        START_TYPE=0,
        END_TYPE=1,
        hs=128,
        nz=32,
        bidirectional=False
    )
    
    # Create dummy batch
    batch_size = 4
    max_n = 20
    nvt = 6
    xs = nvt + max_n - 1
    
    # Random graph data
    G = torch.randn(batch_size, max_n - 1, xs)
    
    # Set first nvt dimensions to valid one-hot (simulate node types)
    G[:, :, :nvt] = 0
    for b in range(batch_size):
        for i in range(max_n - 1):
            node_type = torch.randint(0, nvt, (1,)).item()
            G[b, i, node_type] = 1.0
    
    # Forward pass
    (node_logits, edge_logits), mu, logvar = model(G)
    
    print(f"✓ Forward pass successful")
    print(f"  Node logits shape: {node_logits.shape}")
    print(f"  Edge logits shape: {edge_logits.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Logvar shape: {logvar.shape}")


def test_loss():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    model = DVAE_AIG(
        max_n=20,
        nvt=6,
        START_TYPE=0,
        END_TYPE=1,
        hs=128,
        nz=32,
        bidirectional=False
    )
    
    # Create dummy batch
    batch_size = 4
    max_n = 20
    nvt = 6
    xs = nvt + max_n - 1
    
    G = torch.zeros(batch_size, max_n - 1, xs)
    
    # Set valid one-hot node types
    for b in range(batch_size):
        for i in range(max_n - 1):
            node_type = torch.randint(2, nvt, (1,)).item()  # Skip START and END
            G[b, i, node_type] = 1.0
    
    # Compute loss
    total_loss, recon_loss, kl_loss = model.loss(G, beta=1.0)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Recon loss: {recon_loss.item():.4f}")
    print(f"  KL loss: {kl_loss.item():.4f}")


def test_generation():
    """Test graph generation."""
    print("\nTesting generation...")
    
    model = DVAE_AIG(
        max_n=20,
        nvt=6,
        START_TYPE=0,
        END_TYPE=1,
        hs=128,
        nz=32,
        bidirectional=False
    )
    model.eval()
    
    # Generate samples
    num_samples = 3
    graphs = model.generate(num_samples)
    
    print(f"✓ Generated {len(graphs)} graphs")
    for i, g in enumerate(graphs):
        print(f"  Graph {i}: {g['num_nodes']} nodes, {len(g['edges'])} edges")


def test_bidirectional():
    """Test bidirectional model."""
    print("\nTesting bidirectional model...")
    
    model = DVAE_AIG(
        max_n=20,
        nvt=6,
        START_TYPE=0,
        END_TYPE=1,
        hs=128,
        nz=32,
        bidirectional=True
    )
    
    batch_size = 2
    max_n = 20
    nvt = 6
    xs = nvt + max_n - 1
    
    G = torch.zeros(batch_size, max_n - 1, xs)
    for b in range(batch_size):
        for i in range(max_n - 1):
            node_type = torch.randint(2, nvt, (1,)).item()
            G[b, i, node_type] = 1.0
    
    total_loss, recon_loss, kl_loss = model.loss(G, beta=1.0)
    
    print(f"✓ Bidirectional model works")
    print(f"  Total loss: {total_loss.item():.4f}")


def main():
    print("=" * 60)
    print("D-VAE Baseline Quick Test")
    print("=" * 60)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_loss()
        test_generation()
        test_bidirectional()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
