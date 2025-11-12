# D-VAE Baseline for AIG Generation

This directory implements D-VAE (DAG Variational Autoencoder) as a baseline for AND-Inverter Graph (AIG) generation.

## Overview

D-VAE is a variational autoencoder designed for directed acyclic graphs (DAGs). It uses asynchronous message passing to encode/decode graphs, where a node updates its state only after all predecessors have been updated. This allows the model to encode the computation on a DAG, not just local structure.

**Original Paper**: "D-VAE: A Variational Autoencoder for Directed Acyclic Graphs" by Zhang et al., NeurIPS 2019  
**Original Implementation**: https://github.com/muhanzhang/D-VAE

## Architecture

The D-VAE implementation for AIGs consists of:

1. **Encoder**: GRU-based encoder that processes sequential node representations
   - Input: One-hot node type + binary vector of connections from predecessors
   - Output: Latent distribution parameters (μ, log σ²)

2. **Latent Space**: Continuous latent representation via reparameterization trick
   - Enables gradient-based optimization
   - Allows generation by sampling from N(0, I)

3. **Decoder**: GRU-based decoder that generates graphs step-by-step
   - Predicts node types and edge connections sequentially
   - Uses teacher forcing during training

## Files

- `model.py`: Core D-VAE model architecture
- `data_loader.py`: Data loading and preprocessing for AIG graphs
- `train.py`: Training script
- `sample.py`: Sampling/generation script
- `config.yaml`: Default configuration

## Usage

### Training

```bash
python -m aig2pt.baselines.dvae.train \
    --data-dir /path/to/processed/data \
    --output-dir ./dvae_checkpoints \
    --max-n 56 \
    --nvt 6 \
    --hidden-size 256 \
    --latent-size 56 \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-4
```

### Sampling

```bash
python -m aig2pt.baselines.dvae.sample \
    --checkpoint ./dvae_checkpoints/best_model.pt \
    --output ./generated_graphs.json \
    --num-samples 1000 \
    --temperature 1.0
```

## Configuration

### Model Parameters

- `max_n`: Maximum number of nodes (default: 56)
- `nvt`: Number of vertex types including START/END tokens (default: 6)
  - 0: START (padding)
  - 1: END
  - 2: CONST0
  - 3: PI (Primary Input)
  - 4: AND
  - 5: PO (Primary Output)
- `hidden_size`: GRU hidden size (default: 256)
- `latent_size`: Dimension of latent space (default: 56)
- `bidirectional`: Use bidirectional encoder (default: False)

### Training Parameters

- `batch_size`: Training batch size (default: 32)
- `lr`: Learning rate (default: 1e-4)
- `beta`: Weight for KL divergence term (default: 1.0)
- `eps_scale`: Temperature for sampling during training (default: 1.0)
- `epochs`: Number of training epochs (default: 100)

## Data Format

D-VAE expects graphs in sequential format where each node is represented as:
```
[one_hot(node_type), binary_vector(connections_from_predecessors)]
```

The data loader automatically converts PyTorch Geometric Data objects to this format.

## Evaluation

The generated graphs can be evaluated using the same metrics as AIG2PT:

- **Validity**: Percentage of generated graphs that are valid AIGs
- **Uniqueness**: Percentage of unique graphs among valid ones
- **Novelty**: Percentage of valid unique graphs not in training set

Use the evaluation utilities in `aig2pt/evaluate.py` for consistent comparison.

## Comparison with AIG2PT

| Aspect | D-VAE | AIG2PT |
|--------|-------|--------|
| Architecture | VAE with GRU encoder/decoder | GPT-style transformer |
| Latent Space | Continuous (Gaussian) | Discrete (next-token prediction) |
| Generation | Sample from latent → decode | Autoregressive sampling |
| Training | Reconstruction + KL divergence | Next-token prediction |
| Complexity | O(n²) for n nodes | O(n·k²) for sequence length n, context k |

## Citation

If you use this D-VAE baseline, please cite both the original D-VAE paper and AIG2PT:

```bibtex
@inproceedings{zhang2019dvae,
  title={D-VAE: A Variational Autoencoder for Directed Acyclic Graphs},
  author={Zhang, Muhan and Jiang, Shali and Cui, Zhicheng and Garnett, Roman and Chen, Yixin},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1586--1598},
  year={2019}
}
```

## Notes

- The implementation uses teacher forcing during training for stable convergence
- Latent space interpolation is possible due to continuous representation
- The model can be extended with property prediction heads for downstream tasks
- For best results, tune β (KL weight) and learning rate on validation set
