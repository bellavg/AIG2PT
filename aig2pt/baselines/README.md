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
