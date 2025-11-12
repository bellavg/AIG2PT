"""
D-VAE baseline for AIG generation.

This module implements D-VAE (DAG Variational Autoencoder) adapted for 
AND-Inverter Graph (AIG) generation as a baseline to compare against AIG2PT.
"""

# Import model without dependencies
from .model import DVAE_AIG

# Other imports will be done lazily when needed
__all__ = ['DVAE_AIG']
"""D-VAE baseline module."""

from .model import DVAEBaseline

__all__ = ['DVAEBaseline']
