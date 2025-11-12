"""Setup utilities for LayerDAG baseline."""
import random
import numpy as np
import torch
import yaml


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(file_path):
    """Load YAML configuration file."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
