"""
Example: Using LayerDAG Baseline with AIG2PT

This script demonstrates how to integrate the LayerDAG baseline
with the existing aig2pt framework for unconditional AIG generation.
"""
import os
import sys

# Example usage with aig2pt dataset
def example_layerdag_training():
    """Example of training LayerDAG on AIG dataset."""
    
    print("="*60)
    print("LayerDAG Training Example")
    print("="*60)
    
    # Typical data directory structure for aig2pt
    # Assumes you have PyG data in this format
    data_dir = "/path/to/your/aig/dataset"
    
    print(f"\n1. Data directory: {data_dir}")
    print("   Expected structure:")
    print("     raw/")
    print("       train.pt  (list of PyG Data objects)")
    print("       val.pt")
    print("       test.pt")
    
    print("\n2. Loading dataset...")
    # In practice, uncomment below to actually run:
    # from aig2pt.baselines.layerdag.dataset import load_aig_dataset
    # train_set, val_set, test_set = load_aig_dataset(data_dir)
    print("   from aig2pt.baselines.layerdag.dataset import load_aig_dataset")
    print("   train_set, val_set, test_set = load_aig_dataset(data_dir)")
    
    print("\n3. Training Stage 1 (Node Count Prediction)...")
    print("   cd aig2pt/baselines/layerdag")
    print("   python train_layerdag.py \\")
    print("       --config_file configs/aig.yaml \\")
    print(f"       --data_dir {data_dir} \\")
    print("       --output_dir ./checkpoints \\")
    print("       --seed 42")
    
    print("\n4. Expected output:")
    print("   checkpoints/node_count_model_YYYYMMDD_HHMMSS.pth")
    
    print("\n" + "="*60)


def example_comparison_with_gpt():
    """Example comparing LayerDAG with GPT baseline."""
    
    print("="*60)
    print("Comparing LayerDAG vs GPT for AIG Generation")
    print("="*60)
    
    print("\nGPT Approach (aig2pt current model):")
    print("  - Sequential token generation")
    print("  - Learns: <boc> NODE_0 NODE_1 ... <eoc> <bog> EDGE_0 EDGE_1 ... <eog>")
    print("  - Pros: Simple, flexible, can condition on prefixes")
    print("  - Cons: May struggle with long-range dependencies")
    
    print("\nLayerDAG Approach (baseline):")
    print("  - Layer-wise generation")
    print("  - Stage 1: Predict layer sizes")
    print("  - Stage 2: Generate node types per layer (diffusion)")
    print("  - Stage 3: Generate edges between layers (diffusion)")
    print("  - Pros: Explicit DAG structure, handles layers naturally")
    print("  - Cons: More complex, requires 3 separate models")
    
    print("\nEvaluation Metrics (V.U.N.):")
    print("  - Validity: % of generated AIGs that are valid DAGs")
    print("  - Uniqueness: % of generated AIGs that are unique")
    print("  - Novelty: % of generated AIGs not in training set")
    
    print("\n" + "="*60)


def example_dataset_statistics():
    """Example of analyzing AIG dataset statistics for LayerDAG."""
    
    print("="*60)
    print("AIG Dataset Statistics for LayerDAG")
    print("="*60)
    
    print("\nKey statistics to collect:")
    print("  1. Layer distribution:")
    print("     - Number of layers per graph")
    print("     - Nodes per layer (mean, std, max)")
    
    print("\n  2. Node type distribution:")
    print("     - CONST: constant nodes")
    print("     - PI: primary inputs")
    print("     - AND: AND gates")
    
    print("\n  3. Edge statistics:")
    print("     - Forward edges (FWD)")
    print("     - Inverting edges (INV)")
    print("     - In-degree / Out-degree distributions")
    
    print("\n  4. Graph-level statistics:")
    print("     - Total nodes per graph")
    print("     - Total edges per graph")
    print("     - Depth (max layer)")
    
    print("\nThese statistics help tune LayerDAG hyperparameters:")
    print("  - Max layer size determines model capacity")
    print("  - Node type distribution affects embedding sizes")
    print("  - Edge distribution affects diffusion steps")
    
    print("\n" + "="*60)


def example_end_to_end_workflow():
    """Complete workflow example."""
    
    print("="*60)
    print("End-to-End LayerDAG Workflow for AIG Generation")
    print("="*60)
    
    workflow = """
Step 1: Prepare Data
  → Convert .aig files to PyG format
  → Save as raw/train.pt, raw/val.pt, raw/test.pt
  
Step 2: Analyze Dataset
  → Compute statistics (layers, nodes, edges)
  → Determine hyperparameters
  
Step 3: Train Stage 1 (Node Count)
  → python train_layerdag.py --stage node_count
  → Saves: checkpoints/node_count_model.pth
  
Step 4: Train Stage 2 (Node Diffusion) [TO BE IMPLEMENTED]
  → python train_layerdag.py --stage node_diffusion
  → Saves: checkpoints/node_diffusion_model.pth
  
Step 5: Train Stage 3 (Edge Diffusion) [TO BE IMPLEMENTED]
  → python train_layerdag.py --stage edge_diffusion
  → Saves: checkpoints/edge_diffusion_model.pth
  
Step 6: Sample AIGs
  → python sample_layerdag.py --num_samples 1000
  → Generates: samples/sampled_aigs.pth
  
Step 7: Evaluate
  → Compute V.U.N. metrics
  → Compare with GPT baseline
  → Analyze graph properties
"""
    
    print(workflow)
    print("="*60)


def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print(" " * 15 + "LayerDAG Baseline Integration Examples")
    print("="*70 + "\n")
    
    example_layerdag_training()
    print("\n")
    
    example_comparison_with_gpt()
    print("\n")
    
    example_dataset_statistics()
    print("\n")
    
    example_end_to_end_workflow()
    
    print("\n" + "="*70)
    print("For more details, see:")
    print("  - aig2pt/baselines/layerdag/README.md")
    print("  - LAYERDAG_QUICKSTART.md")
    print("  - LayerDAG paper: https://arxiv.org/abs/2411.02322")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
