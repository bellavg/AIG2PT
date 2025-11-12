"""
Base interface for baseline models.

All baseline models should inherit from BaselineModel and implement
the required methods for unconditional AIG generation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import numpy as np


class BaselineModel(ABC):
    """
    Abstract base class for baseline AIG generation models.
    
    All baseline models must implement:
    - load_pretrained: Load a pretrained model or initialize from scratch
    - generate: Generate AIGs unconditionally
    - save_model: Save the model state
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the baseline model.
        
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def load_pretrained(self, checkpoint_path: Optional[str] = None):
        """
        Load a pretrained model or initialize from scratch.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, initialize randomly.
        """
        pass
    
    @abstractmethod
    def generate(self, 
                 num_samples: int = 1,
                 max_nodes: int = 50,
                 temperature: float = 1.0,
                 **kwargs) -> List[Dict[str, Any]]:
        """
        Generate AIGs unconditionally.
        
        Args:
            num_samples: Number of AIGs to generate
            max_nodes: Maximum number of nodes in generated AIGs
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated AIGs, each as a dictionary with:
                - 'nodes': List of node types
                - 'edges': List of edge connections
                - 'edge_types': List of edge types (regular/inverted)
        """
        pass
    
    @abstractmethod
    def save_model(self, save_path: str):
        """
        Save the model state.
        
        Args:
            save_path: Path where to save the model
        """
        pass
    
    def to_aig_format(self, generated_data: Any) -> Dict[str, Any]:
        """
        Convert model-specific output to standard AIG format.
        
        Args:
            generated_data: Model-specific generated output
            
        Returns:
            AIG in standard format with nodes, edges, and edge_types
        """
        # Default implementation - can be overridden by subclasses
        return generated_data
    
    def train(self, train_data, valid_data=None, **kwargs):
        """
        Train the model (optional - for models that need training).
        
        Args:
            train_data: Training dataset
            valid_data: Validation dataset
            **kwargs: Additional training parameters
        """
        raise NotImplementedError("This baseline does not support training")
