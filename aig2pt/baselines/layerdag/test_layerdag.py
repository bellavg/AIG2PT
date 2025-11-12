"""
Basic tests for LayerDAG baseline.

These tests verify that the dataset conversion and model initialization work correctly.
"""
import sys
import os
import torch
from torch_geometric.data import Data

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import AIGDAGDataset, load_aig_dataset
from dataset.layer_dag import LayerDAGNodeCountDataset


def test_aig_dataset_creation():
    """Test that we can create an AIG dataset."""
    print("Testing AIG dataset creation...")
    
    dataset = AIGDAGDataset(num_categories=3, label=False)
    assert dataset.num_categories == 3
    assert not dataset.conditional
    print("✓ AIG dataset created successfully")


def test_aig_data_addition():
    """Test adding AIG data from PyG format."""
    print("\nTesting AIG data addition...")
    
    # Create a simple AIG: 2 PIs connected to 1 AND gate
    # Node types: 0=CONST, 1=PI, 2=AND
    x_n = torch.tensor([1, 1, 2])  # PI, PI, AND
    edge_index = torch.tensor([
        [0, 1],  # Source nodes
        [2, 2]   # Destination nodes (both to AND gate)
    ])
    
    pyg_data = Data(x=x_n, edge_index=edge_index)
    
    dataset = AIGDAGDataset(num_categories=3, label=False)
    dataset.add_aig_data(pyg_data)
    
    assert len(dataset) == 1
    print(f"✓ Added AIG with {len(x_n)} nodes and {edge_index.shape[1]} edges")
    
    # Retrieve the data
    if dataset.conditional:
        src, dst, x, y = dataset[0]
    else:
        src, dst, x = dataset[0]
    
    print(f"  Retrieved: {len(x)} nodes, {len(src)} edges")
    print("✓ Data addition successful")


def test_layerdag_dataset_conversion():
    """Test converting to LayerDAG node count dataset."""
    print("\nTesting LayerDAG dataset conversion...")
    
    # Create a simple AIG dataset
    dataset = AIGDAGDataset(num_categories=3, label=False)
    
    # Add a simple graph
    x_n = torch.tensor([1, 1, 2])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    pyg_data = Data(x=x_n, edge_index=edge_index)
    dataset.add_aig_data(pyg_data)
    
    # Convert to LayerDAG node count dataset
    node_count_dataset = LayerDAGNodeCountDataset(dataset, conditional=False)
    
    print(f"✓ Created LayerDAG node count dataset with {len(node_count_dataset)} examples")


def test_one_hot_node_features():
    """Test handling one-hot encoded node features."""
    print("\nTesting one-hot encoded node features...")
    
    # Create AIG with one-hot encoded features
    # 3 nodes: [1,0,0], [1,0,0], [0,0,1] = PI, PI, AND
    x_n = torch.tensor([
        [0, 1, 0],  # PI
        [0, 1, 0],  # PI
        [0, 0, 1]   # AND
    ], dtype=torch.float)
    
    edge_index = torch.tensor([[0, 1], [2, 2]])
    pyg_data = Data(x=x_n, edge_index=edge_index)
    
    dataset = AIGDAGDataset(num_categories=3, label=False)
    dataset.add_aig_data(pyg_data)
    
    assert len(dataset) == 1
    print("✓ One-hot encoded features handled correctly")


def test_empty_dataset():
    """Test handling empty dataset."""
    print("\nTesting empty dataset handling...")
    
    dataset = AIGDAGDataset(num_categories=3, label=False)
    assert len(dataset) == 0
    print("✓ Empty dataset handled correctly")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running LayerDAG Baseline Tests")
    print("="*60)
    
    try:
        test_aig_dataset_creation()
        test_aig_data_addition()
        test_layerdag_dataset_conversion()
        test_one_hot_node_features()
        test_empty_dataset()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed with error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
