# LayerDAG Baseline Quick Start

This guide helps you get started with the LayerDAG baseline for unconditional AIG generation.

## What is LayerDAG?

LayerDAG is a state-of-the-art baseline model for generating directed acyclic graphs (DAGs). Unlike sequential token-based models like GPT, LayerDAG generates graphs layer-by-layer using a combination of:
- **Node count prediction**: Predicts how many nodes in each layer
- **Node type diffusion**: Generates node types using discrete diffusion
- **Edge diffusion**: Generates edges between layers

## Installation

### Additional Dependencies

LayerDAG requires dependencies beyond the base aig2pt environment:

```bash
# DGL (Deep Graph Library) - required for graph operations
pip install dgl

# Other dependencies
pip install einops pydantic pyyaml tqdm
```

**Note**: If using CUDA, install the CUDA-compatible version:
```bash
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
```

## Quick Usage

### 1. Prepare Your Data

AIGs start as raw `.aig` files in AIGER format. Organize them by split:

```
raw_aig_files/
  train/
    circuit1.aig
    circuit2.aig
    ...
  val/
    ...
  test/
    ...
```

### 2. Preprocess to PyG Format

Convert raw `.aig` files to PyTorch Geometric format:

```bash
cd aig2pt/baselines/layerdag

python preprocess_aigs.py \
    --input_dir /path/to/raw_aig_files \
    --output_dir /path/to/processed_data
```

This creates `processed_data/raw/{train,val,test}.pt` files.

### 3. Train the Model

```bash
cd aig2pt/baselines/layerdag

python train_layerdag.py \
    --config_file configs/aig.yaml \
    --data_dir /path/to/processed_data \
    --output_dir ./checkpoints \
    --seed 42
```

### 4. Generate AIGs

```bash
python sample_layerdag.py \
    --model_path checkpoints/node_count_model_*.pth \
    --num_samples 100 \
    --output_dir ./samples
```

## Current Implementation Status

**Implemented:**
- ✅ Dataset adapter for AIG PyG format
- ✅ Node count prediction (Stage 1 of 3)
- ✅ Training infrastructure
- ✅ Configuration management

**To Be Implemented:**
- ⚠️ Node type diffusion (Stage 2 of 3)
- ⚠️ Edge diffusion (Stage 3 of 3)
- ⚠️ Complete sampling pipeline

## Why Use LayerDAG?

LayerDAG offers several advantages as a baseline:

1. **Different Approach**: Uses layer-wise generation instead of sequential tokens
2. **Structural Awareness**: Explicitly models DAG layer structure
3. **Proven Performance**: State-of-the-art results on DAG generation benchmarks
4. **Comparison**: Provides a strong baseline to compare against GPT-based approaches

## Directory Structure

```
aig2pt/baselines/layerdag/
├── configs/              # Configuration files
│   └── aig.yaml
├── dataset/              # Dataset adapters
│   ├── aig_layerdag.py   # AIG-specific adapter
│   ├── general.py        # Base DAG dataset
│   └── layer_dag.py      # LayerDAG datasets
├── model/                # Model components
│   ├── layer_dag.py      # Core model
│   └── diffusion.py      # Diffusion models
├── train_layerdag.py     # Training script
├── sample_layerdag.py    # Sampling script
└── README.md             # Detailed documentation
```

## For More Information

See the comprehensive README in `aig2pt/baselines/layerdag/README.md` for:
- Detailed installation instructions
- Configuration options
- Implementation details
- Troubleshooting
- References to the original paper

## Citation

If you use LayerDAG in your research, please cite:

```bibtex
@inproceedings{li2024layerdag,
    title={LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation},
    author={Mufei Li and Viraj Shitole and Eli Chien and Changhai Man and Zhaodong Wang and Srinivas Sridharan and Ying Zhang and Tushar Krishna and Pan Li},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```

Original repository: https://github.com/Graph-COM/LayerDAG
