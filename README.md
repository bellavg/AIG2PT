# AIG2PT

Foundation model for AND-Inverter Graph (AIG) generation and analysis.

## Features

- **GPT-based AIG Generation**: Autoregressive transformer model for AIG generation
- **Baselines**: Multiple baseline models for comparison
  - **LayerDAG**: Layer-wise DAG generation using discrete diffusion ([Quick Start](LAYERDAG_QUICKSTART.md))
- **Evaluation Metrics**: V.U.N. (Validity, Uniqueness, Novelty) metrics
- **Dataset Processing**: Tools for preparing and processing AIG datasets

## Quick Start

See [LAYERDAG_QUICKSTART.md](LAYERDAG_QUICKSTART.md) for getting started with the LayerDAG baseline.

## Project Structure

```
AIG2PT/
├── aig2pt/                    # Main source code
│   ├── baselines/            # Baseline models
│   │   └── layerdag/         # LayerDAG baseline
│   ├── core/                 # Core model (GPT)
│   ├── dataset/              # Dataset utilities
│   └── configs/              # Configuration files
├── LAYERDAG_QUICKSTART.md    # LayerDAG quick start guide
└── plan.md                   # Development plan
```

## Baselines

### LayerDAG

A layer-wise autoregressive diffusion model for DAG generation. See [aig2pt/baselines/layerdag/README.md](aig2pt/baselines/layerdag/README.md) for details.

## License

See LICENSE files for details.