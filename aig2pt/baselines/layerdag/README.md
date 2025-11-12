# LayerDAG Baseline for AIG Unconditional Generation

This directory contains an adaptation of the LayerDAG model for unconditional AND-Inverter Graph (AIG) generation. LayerDAG is a layerwise autoregressive diffusion model that generates directed acyclic graphs (DAGs) layer by layer.

## Overview

LayerDAG generates DAGs through a three-stage process:
1. **Node Count Prediction**: Predicts the number of nodes in each layer
2. **Node Type Diffusion**: Generates node types using discrete diffusion
3. **Edge Diffusion**: Generates edges between layers using discrete diffusion

This implementation adapts LayerDAG to work with AIG-specific constraints:
- Node types: CONST (constant 0), PI (primary input), AND (AND gate)
- Edge types: FWD (forward/non-inverting), INV (inverting)
- DAG structure with topological ordering

## Installation

### Prerequisites

LayerDAG requires additional dependencies beyond the base aig2pt environment:

```bash
# Install PyTorch (if not already installed)
pip install torch==1.12.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install DGL (Deep Graph Library)
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html

# Install other dependencies
pip install tqdm einops pydantic pyyaml
```

**Note**: If you encounter errors about `libcusparse.so`, you may need to set the library path:
```bash
export LD_LIBRARY_PATH=/path/to/conda/envs/aig2pt/lib:$LD_LIBRARY_PATH
```

## Data Format

LayerDAG expects AIGs in PyTorch Geometric format. The dataset should be organized as:

```
data_dir/
  raw/
    train.pt  # List of PyG Data objects
    val.pt
    test.pt
```

Each PyG Data object should have:
- `x`: Node features (shape: [num_nodes] or [num_nodes, num_features])
- `edge_index`: Edge connectivity (shape: [2, num_edges])
- `edge_attr`: Edge attributes (optional, for edge inversions)

## Training

### Basic Training

Train the LayerDAG model on your AIG dataset:

```bash
cd aig2pt/baselines/layerdag

python train_layerdag.py \
    --config_file configs/aig.yaml \
    --data_dir /path/to/aig/dataset \
    --output_dir ./checkpoints \
    --seed 42
```

### Configuration

Edit `configs/aig.yaml` to customize training hyperparameters:

```yaml
general:
  dataset: aig
  conditional: false  # Set to true for conditional generation
  patience: 10        # Early stopping patience

node_count:
  loader:
    batch_size: 64
    num_workers: 2
  model:
    x_n_emb_size: 64
    num_mpnn_layers: 3
  num_epochs: 300
  optimizer:
    lr: 0.0003

# ... (node_pred and edge_pred configurations)
```

### Training Stages

**Current Implementation**: The training script implements **Stage 1 (Node Count Prediction)** only.

**Full LayerDAG Training** requires three stages:
1. ✅ Node Count Prediction (implemented)
2. ⚠️ Node Type Diffusion (to be implemented)
3. ⚠️ Edge Diffusion (to be implemented)

To train the complete model, you'll need to:
1. Train the node count model (current script)
2. Train the node diffusion model (requires extending the script)
3. Train the edge diffusion model (requires extending the script)

## Sampling

Generate synthetic AIGs using a trained model:

```bash
python sample_layerdag.py \
    --model_path checkpoints/node_count_model_*.pth \
    --num_samples 100 \
    --output_dir ./samples \
    --seed 42
```

**Note**: Full sampling requires all three trained models (node count, node diffusion, edge diffusion).

## Directory Structure

```
layerdag/
├── configs/
│   └── aig.yaml              # Training configuration
├── dataset/
│   ├── __init__.py
│   ├── aig_layerdag.py       # AIG dataset adapter
│   ├── general.py            # Base DAG dataset (from LayerDAG)
│   └── layer_dag.py          # LayerDAG dataset utilities
├── model/
│   ├── __init__.py
│   ├── layer_dag.py          # LayerDAG model architecture
│   └── diffusion.py          # Discrete diffusion models
├── train_layerdag.py         # Training script
├── sample_layerdag.py        # Sampling script
├── setup_utils.py            # Utility functions
└── README.md                 # This file
```

## Implementation Status

### Completed ✅
- [x] Dataset adapter for AIG PyG format
- [x] Configuration file for AIG generation
- [x] Node count prediction training
- [x] Basic project structure
- [x] Documentation

### To Be Implemented ⚠️
- [ ] Node type diffusion training
- [ ] Edge diffusion training
- [ ] Full sampling pipeline
- [ ] Evaluation metrics (V.U.N. - Validity, Uniqueness, Novelty)
- [ ] Integration with aig2pt evaluation framework

## Usage Example

```python
from aig2pt.baselines.layerdag.dataset import load_aig_dataset
from aig2pt.baselines.layerdag.dataset.layer_dag import LayerDAGNodeCountDataset

# Load AIG dataset
train_set, val_set, test_set = load_aig_dataset('/path/to/data')

# Create LayerDAG dataset for node count prediction
train_node_count = LayerDAGNodeCountDataset(train_set, conditional=False)

print(f"Loaded {len(train_node_count)} training examples")
```

## Evaluation

To evaluate generated AIGs, you can use the aig2pt evaluation framework:

```python
from aig2pt.sampling_and_evaluation import compute_vun_metrics

# Load generated AIGs
generated_aigs = torch.load('samples/sampled_aigs.pth')

# Compute V.U.N. metrics
metrics = compute_vun_metrics(generated_aigs, reference_aigs)
print(f"Validity: {metrics['validity']:.2%}")
print(f"Uniqueness: {metrics['uniqueness']:.2%}")
print(f"Novelty: {metrics['novelty']:.2%}")
```

## Reference

LayerDAG is based on the paper:

```bibtex
@inproceedings{li2024layerdag,
    title={LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation},
    author={Mufei Li and Viraj Shitole and Eli Chien and Changhai Man and Zhaodong Wang and Srinivas Sridharan and Ying Zhang and Tushar Krishna and Pan Li},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```

Original LayerDAG repository: https://github.com/Graph-COM/LayerDAG

## Limitations and Future Work

1. **Incomplete Implementation**: This is a minimal baseline implementation. The full LayerDAG pipeline includes node and edge diffusion stages that need to be implemented for complete unconditional generation.

2. **AIG-Specific Constraints**: The current implementation doesn't enforce all AIG structural constraints (e.g., AND gates must have exactly 2 inputs). These should be added in the diffusion models.

3. **Edge Inversions**: Handling edge inversions (NOT gates) as edge attributes requires careful integration with the edge diffusion model.

4. **Performance**: The model has not been tuned for AIG generation specifically. Hyperparameter optimization may be needed.

## Contributing

To extend this baseline:

1. Implement node type diffusion training in `train_layerdag.py`
2. Implement edge diffusion training in `train_layerdag.py`
3. Complete the sampling pipeline in `sample_layerdag.py`
4. Add AIG-specific validation and metrics
5. Tune hyperparameters for optimal AIG generation

## Support

For issues specific to this LayerDAG adaptation, please refer to the main aig2pt documentation.

For issues with the original LayerDAG implementation, see: https://github.com/Graph-COM/LayerDAG
