# Baselines for AIG2PT

This directory contains baseline model implementations for comparison with the main AIG2PT model.

## Available Baselines

### D-VAE (DAG Variational Autoencoder)

D-VAE is a variational autoencoder specifically designed for directed acyclic graphs (DAGs). It uses asynchronous message passing to encode the computation structure of graphs.

**Directory**: `dvae/`

**Key Features**:
- Continuous latent space representation
- Autoregressive graph generation
- Support for unconditional generation
- Compatible with AIG data format

**Quick Start**:
```bash
# Training
python -m aig2pt.baselines.dvae.train \
    --data-dir /path/to/processed/data \
    --output-dir ./dvae_checkpoints \
    --epochs 100

# Sampling
python -m aig2pt.baselines.dvae.sample \
    --checkpoint ./dvae_checkpoints/best_model.pt \
    --output ./generated_graphs.json \
    --num-samples 1000
```

See [dvae/README.md](dvae/README.md) for detailed documentation.

## Adding New Baselines

To add a new baseline model:

1. Create a new directory under `baselines/`
2. Implement the following components:
   - `model.py`: Core model architecture
   - `data_loader.py`: Data preprocessing and loading
   - `train.py`: Training script
   - `sample.py`: Generation/sampling script
   - `README.md`: Documentation
   - `config.yaml`: Default configuration

3. Ensure compatibility with AIG data format
4. Follow the same evaluation metrics (V.U.N.: Validity, Uniqueness, Novelty)

## Comparison Framework

All baselines should support:

### Input Format
- PyTorch Geometric Data objects from `aig2pt/dataset/`
- Conversion utilities for AIG-specific format

### Output Format
- Generated graphs in AIG format
- JSON export for analysis
- Compatibility with evaluation scripts

### Evaluation Metrics
- **Validity**: % of generated graphs that are valid AIGs
- **Uniqueness**: % of unique graphs among valid ones  
- **Novelty**: % of valid unique graphs not seen in training
- **Distribution metrics**: Graph size, node/edge distributions

### Performance Metrics
- Training time
- Inference time
- Model size (parameters)
- Memory usage

## Baseline Comparison Table

| Model | Architecture | Latent Space | Generation | Parameters | Status |
|-------|--------------|--------------|------------|------------|--------|
| **AIG2PT** | GPT Transformer | Discrete (tokens) | Autoregressive | ~50M | Main |
| **D-VAE** | GRU VAE | Continuous (Gaussian) | Decoder from latent | ~170K | ✅ Ready |
| LayerDAG | TBD | TBD | TBD | TBD | 📝 Planned |
| GraphRNN | TBD | TBD | TBD | TBD | 📝 Planned |

## Citation

If you use these baselines in your research, please cite the original papers along with AIG2PT:

### D-VAE
```bibtex
@inproceedings{zhang2019dvae,
  title={D-VAE: A Variational Autoencoder for Directed Acyclic Graphs},
  author={Zhang, Muhan and Jiang, Shali and Cui, Zhicheng and Garnett, Roman and Chen, Yixin},
  booktitle={NeurIPS},
  year={2019}
}
```

## Contributing

To contribute a new baseline:
1. Fork the repository
2. Implement the baseline following the structure above
3. Add comprehensive documentation
4. Include tests and example usage
5. Submit a pull request

## License

See individual baseline directories for specific license information.
# Baseline Models for AIG Generation

This directory contains adapters for baseline models to work with AND-Inverter Graph (AIG) generation for comparison with the AIG2PT model.

## Available Baselines

### 1. Circuit Transformer
- **Source**: [snowkylin/circuit-transformer](https://github.com/snowkylin/circuit-transformer)
- **Paper**: "Circuit Transformer: End-to-end Circuit Design by Predicting the Next Gate" (ICLR 2025)
- **Description**: Transformer-based model with cutoff properties that ensures logical equivalence preservation during circuit generation.
- **Key Features**:
  - Stepwise generation with validity checking
  - Preserves logical equivalence
  - Optimized for AIG representation

### 2. LayerDAG
- **Source**: [Graph-COM/LayerDAG](https://github.com/Graph-COM/LayerDAG)
- **Paper**: "LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graphs" (ICLR 2025)
- **Description**: Layerwise autoregressive diffusion model specifically designed for DAG generation.
- **Key Features**:
  - Layer-wise decomposition of DAGs
  - Autoregressive generation
  - Diffusion modeling for dependencies

### 3. D-VAE
- **Source**: D-VAE: A Variational Autoencoder for Directed Acyclic Graphs (NeurIPS 2019)
- **Paper**: [D-VAE Paper](https://papers.neurips.cc/paper/8437-d-vae-a-variational-autoencoder-for-directed-acyclic-graphs.pdf)
- **Description**: Variational Autoencoder architecture designed for DAGs with asynchronous message passing.
- **Key Features**:
  - VAE architecture for DAGs
  - Asynchronous message passing in encoder
  - Sequential decoder for generation

## Usage

### Quick Start

Generate AIGs using any baseline model:

```bash
# Circuit Transformer
python run_baseline.py --baseline circuit_transformer --num_samples 100

# LayerDAG
python run_baseline.py --baseline layerdag --num_samples 100

# D-VAE
python run_baseline.py --baseline dvae --num_samples 100
```

### Command-Line Arguments

```
--baseline          Baseline model to use (circuit_transformer, layerdag, dvae)
--num_samples       Number of AIGs to generate (default: 100)
--max_nodes         Maximum number of AND nodes (default: 50)
--num_inputs        Number of primary inputs (default: 4)
--temperature       Sampling temperature (default: 1.0)
--config            Path to custom config file (optional)
--checkpoint        Path to model checkpoint (optional)
--output            Output directory for results (default: ./baseline_results)
--seed              Random seed (default: 42)
```

### Examples

```bash
# Generate 500 AIGs with Circuit Transformer
python run_baseline.py --baseline circuit_transformer \
    --num_samples 500 \
    --max_nodes 30 \
    --num_inputs 6 \
    --output results/circuit_transformer/

# Generate with custom config
python run_baseline.py --baseline layerdag \
    --config custom_config.yaml \
    --num_samples 200

# Load from checkpoint
python run_baseline.py --baseline dvae \
    --checkpoint checkpoints/dvae_best.pt \
    --num_samples 1000
```

## Output Format

Generated AIGs are saved in JSON format with the following structure:

```json
[
  {
    "nodes": ["CONST0", "PI", "PI", "PI", "PI", "AND", "AND", ...],
    "edges": [[1, 4], [2, 4], [3, 5], [4, 5], ...],
    "edge_types": ["REG", "INV", "REG", "REG", ...],
    "num_inputs": 4,
    "num_and_gates": 10
  },
  ...
]
```

### Node Types
- `CONST0`: Constant zero
- `PI`: Primary Input
- `AND`: AND gate

### Edge Types
- `REG`: Regular (non-inverting) edge
- `INV`: Inverting edge (represents NOT)

## Configuration

Each baseline has a default configuration file in its directory:
- `circuit_transformer/config.yaml`
- `layerdag/config.yaml`
- `dvae/config.yaml`

You can customize these or provide your own config file using the `--config` argument.

### Example Configuration

```yaml
model:
  name: circuit_transformer
  n_embd: 256
  n_layer: 4
  n_head: 4
  max_nodes: 50

generation:
  num_samples: 100
  temperature: 1.0
  max_nodes: 50
  num_inputs: 4
```

## Integration with AIG2PT

These baselines are designed to be compatible with the AIG2PT evaluation framework. The generated AIGs can be directly compared with AIG2PT outputs using the same evaluation metrics.

### Evaluation Metrics

To evaluate generated AIGs:
1. Validity: Percentage of syntactically valid AIGs
2. Uniqueness: Percentage of unique AIGs
3. Novelty: Percentage of AIGs not in training set

## Architecture Details

### Circuit Transformer

The Circuit Transformer adapter uses a simplified transformer architecture:
- Token embeddings for node and edge types
- Positional embeddings
- Multi-head self-attention layers
- Autoregressive generation with validity constraints

### LayerDAG

The LayerDAG adapter implements:
- Layer-wise node organization
- Bipartite edge generation between layers
- Random layer assignment for nodes
- Topological ordering preservation

### D-VAE

The D-VAE adapter implements:
- Graph encoder with mean pooling
- Latent space with reparameterization trick
- Sequential decoder for node and edge generation
- DAG constraint enforcement

## Extending the Baselines

To add a new baseline model:

1. Create a new directory under `baselines/`
2. Implement a class inheriting from `BaselineModel`
3. Implement required methods:
   - `load_pretrained(checkpoint_path)`
   - `generate(num_samples, max_nodes, temperature, ...)`
   - `save_model(save_path)`
4. Add configuration file
5. Register in `BASELINE_MAP` in `run_baseline.py`

Example:

```python
from baselines.base_model import BaselineModel

class MyBaseline(BaselineModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
    
    def load_pretrained(self, checkpoint_path=None):
        # Load or initialize model
        pass
    
    def generate(self, num_samples=1, max_nodes=50, temperature=1.0, **kwargs):
        # Generate AIGs
        return generated_aigs
    
    def save_model(self, save_path):
        # Save model
        pass
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy
- PyYAML

## Notes

- These are simplified implementations demonstrating how to adapt each baseline for AIG generation
- For full functionality, refer to the original implementations
- The models are designed for unconditional generation (no conditioning on specific Boolean functions)
- Random initialization is used by default; provide checkpoints for pretrained models

## References

1. Circuit Transformer: [arXiv:2403.13838](https://arxiv.org/abs/2403.13838)
2. LayerDAG: [arXiv:2411.02322](https://arxiv.org/abs/2411.02322)
3. D-VAE: [NeurIPS 2019](https://papers.neurips.cc/paper/8437-d-vae-a-variational-autoencoder-for-directed-acyclic-graphs.pdf)
