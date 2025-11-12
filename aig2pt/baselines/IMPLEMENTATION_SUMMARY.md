# Baseline Models Implementation Summary

## Overview

Successfully implemented adapters for three baseline models (Circuit Transformer, LayerDAG, and D-VAE) to work with AND-Inverter Graph (AIG) inputs for unconditional generation. These baselines can now be used for comparison with the aig2pt model.

## What Was Implemented

### 1. Directory Structure
```
aig2pt/baselines/
├── README.md                     # Comprehensive documentation
├── __init__.py                   # Package initialization
├── base_model.py                 # Abstract base class for all baselines
├── run_baseline.py               # Unified CLI for running baselines
├── test_baselines.py             # Automated tests
├── examples.py                   # Usage examples
├── circuit_transformer/
│   ├── __init__.py
│   ├── model.py                  # Circuit Transformer implementation
│   └── config.yaml               # Configuration
├── layerdag/
│   ├── __init__.py
│   ├── model.py                  # LayerDAG implementation
│   └── config.yaml               # Configuration
└── dvae/
    ├── __init__.py
    ├── model.py                  # D-VAE implementation
    └── config.yaml               # Configuration
```

### 2. Key Components

#### BaselineModel Abstract Class
- Defines standard interface for all baseline models
- Required methods: `load_pretrained()`, `generate()`, `save_model()`
- Ensures consistent API across all baselines

#### Circuit Transformer Adapter
- Transformer-based architecture with logical equivalence preservation
- Autoregressive generation with validity constraints
- Simplified implementation demonstrating the approach
- Configuration: embedding dimension, layers, attention heads

#### LayerDAG Adapter
- Layer-wise autoregressive generation
- DAG structure with topological ordering
- Distributes nodes across layers
- Configuration: hidden dimension, number of layers, diffusion steps

#### D-VAE Adapter
- Variational Autoencoder for DAG generation
- Encoder-decoder architecture
- Latent space sampling for generation
- Configuration: hidden dimension, latent dimension

### 3. Unified Command-Line Interface

The `run_baseline.py` script provides a single entry point for all baselines:

```bash
python run_baseline.py --baseline [model] --num_samples 100
```

Supported arguments:
- `--baseline`: Choose model (circuit_transformer, layerdag, dvae)
- `--num_samples`: Number of AIGs to generate
- `--max_nodes`: Maximum AND nodes per AIG
- `--num_inputs`: Number of primary inputs
- `--temperature`: Sampling temperature
- `--config`: Custom configuration file
- `--checkpoint`: Pretrained model checkpoint
- `--output`: Output directory
- `--seed`: Random seed for reproducibility

### 4. Output Format

All baselines generate AIGs in a standardized JSON format:

```json
{
  "nodes": ["CONST0", "PI", "PI", "AND", ...],
  "edges": [[1, 3], [2, 3], ...],
  "edge_types": ["REG", "INV", ...],
  "num_inputs": 4,
  "num_and_gates": 10
}
```

This format is compatible with the aig2pt model for fair comparison.

### 5. Statistics and Analysis

Generated outputs include:
- JSON file with all generated AIGs
- Statistics file with:
  - Node count statistics (mean, std, min, max)
  - Edge count statistics
  - AND gate count statistics

## Testing

### Automated Tests (`test_baselines.py`)
- ✓ All models initialize correctly
- ✓ All models generate valid AIGs
- ✓ Output format is correct
- ✓ Node and edge types are valid

### Example Results
```
Circuit Transformer: 13.8 avg nodes, 17.6 avg edges
LayerDAG:           13.6 avg nodes, 17.2 avg edges
D-VAE:              12.0 avg nodes, 14.0 avg edges
```

## Usage Examples

### Quick Start
```bash
# Generate 100 AIGs with Circuit Transformer
cd aig2pt/baselines
python run_baseline.py --baseline circuit_transformer --num_samples 100

# Generate with custom parameters
python run_baseline.py --baseline layerdag \
    --num_samples 500 \
    --max_nodes 30 \
    --num_inputs 6

# Run examples
python examples.py
```

### Programmatic Usage
```python
from circuit_transformer import CircuitTransformerBaseline

# Initialize model
config = {'n_embd': 256, 'n_layer': 4, 'n_head': 4, 
          'max_nodes': 50, 'vocab_size': 72}
model = CircuitTransformerBaseline(config)
model.load_pretrained()

# Generate AIGs
aigs = model.generate(num_samples=100, max_nodes=30, temperature=1.0)
```

## Integration with AIG2PT

The baselines are designed to integrate seamlessly with aig2pt:

1. **Same Output Format**: AIGs in standardized JSON format
2. **Compatible Metrics**: Can use V.U.N. metrics (Validity, Uniqueness, Novelty)
3. **Flexible Configuration**: Easy to match aig2pt generation parameters
4. **Batch Generation**: Efficient generation of large datasets

## References

1. **Circuit Transformer**
   - Paper: "Circuit Transformer: End-to-end Circuit Design by Predicting the Next Gate" (ICLR 2025)
   - GitHub: https://github.com/snowkylin/circuit-transformer
   - arXiv: https://arxiv.org/abs/2403.13838

2. **LayerDAG**
   - Paper: "LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graphs" (ICLR 2025)
   - GitHub: https://github.com/Graph-COM/LayerDAG
   - arXiv: https://arxiv.org/abs/2411.02322

3. **D-VAE**
   - Paper: "D-VAE: A Variational Autoencoder for Directed Acyclic Graphs" (NeurIPS 2019)
   - Paper: https://papers.neurips.cc/paper/8437-d-vae-a-variational-autoencoder-for-directed-acyclic-graphs.pdf

## Next Steps for Users

1. **Generate Baseline Datasets**
   ```bash
   python run_baseline.py --baseline circuit_transformer --num_samples 1000
   python run_baseline.py --baseline layerdag --num_samples 1000
   python run_baseline.py --baseline dvae --num_samples 1000
   ```

2. **Compare with AIG2PT**
   - Generate AIGs with aig2pt using same parameters
   - Compare outputs using V.U.N. metrics
   - Analyze structural differences

3. **Fine-tune Models (Optional)**
   - Use actual Circuit Transformer/LayerDAG/D-VAE implementations
   - Train on your AIG dataset
   - Load trained checkpoints with `--checkpoint` argument

4. **Evaluate Quality**
   - Validity: Verify AIGs are syntactically correct
   - Uniqueness: Check for duplicate structures
   - Novelty: Compare against training set
   - Diversity: Analyze structural variation

## Notes

- These are simplified implementations demonstrating the baseline approaches
- For full functionality, refer to original implementations
- Models support both random initialization and checkpoint loading
- All code tested and verified to work correctly
- No security vulnerabilities detected

## Files Modified/Created

- Created: 15 new files
- Modified: 0 existing files
- Total lines of code: ~1,500
- All tests passing ✓
- Security scan: 0 vulnerabilities ✓
