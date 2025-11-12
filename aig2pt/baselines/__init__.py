"""
Baseline models for AIG generation.

This module provides adapters for various baseline models to work with
AND-Inverter Graph (AIG) inputs for unconditional generation.

Available baselines:
- Circuit Transformer: Transformer-based model with logical equivalence preservation
- LayerDAG: Layerwise autoregressive diffusion model for DAGs
- D-VAE: Variational Autoencoder for directed acyclic graphs
"""

from .base_model import BaselineModel

__all__ = ['BaselineModel']
