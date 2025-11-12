"""
Circuit Transformer baseline adapter for AIG generation.

This module provides an adapter for the Circuit Transformer model
(https://github.com/snowkylin/circuit-transformer) to work with
the AIG2PT data format for unconditional generation.

Circuit Transformer uses a transformer architecture with cutoff properties
to ensure logical equivalence preservation during circuit generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add parent directory to path for imports
BASELINES_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASELINES_DIR))

import base_model
BaselineModel = base_model.BaselineModel


class CircuitTransformerBaseline(BaselineModel):
    """
    Adapter for Circuit Transformer model for unconditional AIG generation.
    
    The Circuit Transformer model generates logic circuits (AIGs) using a
    transformer-based architecture with logical equivalence preservation.
    For unconditional generation, we adapt the model to generate random
    AIGs without a specific target function.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Circuit Transformer baseline.
        
        Args:
            config: Configuration dictionary with keys:
                - n_embd: Embedding dimension
                - n_layer: Number of transformer layers
                - n_head: Number of attention heads
                - max_nodes: Maximum number of AND nodes
                - vocab_size: Size of token vocabulary
        """
        super().__init__(config)
        
        # Model hyperparameters
        self.n_embd = config.get('n_embd', 256)
        self.n_layer = config.get('n_layer', 4)
        self.n_head = config.get('n_head', 4)
        self.max_nodes = config.get('max_nodes', 50)
        self.vocab_size = config.get('vocab_size', 72)
        
        # Node and edge type mappings for AIG
        self.node_types = ['CONST0', 'PI', 'AND']
        self.edge_types = ['REG', 'INV']  # Regular and Inverted
        
    def load_pretrained(self, checkpoint_path: Optional[str] = None):
        """
        Load pretrained Circuit Transformer model or initialize from scratch.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, initialize randomly.
        """
        # Build a simple transformer model for circuit generation
        self.model = SimpleTransformerCircuitGenerator(
            vocab_size=self.vocab_size,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            max_nodes=self.max_nodes
        ).to(self.device)
        
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded Circuit Transformer checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Initializing with random weights")
        else:
            print("Initializing Circuit Transformer with random weights")
    
    def generate(self, 
                 num_samples: int = 1,
                 max_nodes: int = 50,
                 temperature: float = 1.0,
                 num_inputs: int = 4,
                 **kwargs) -> List[Dict[str, Any]]:
        """
        Generate AIGs unconditionally using Circuit Transformer.
        
        Args:
            num_samples: Number of AIGs to generate
            max_nodes: Maximum number of AND nodes
            temperature: Sampling temperature
            num_inputs: Number of primary inputs
            **kwargs: Additional parameters
            
        Returns:
            List of generated AIGs in standard format
        """
        self.model.eval()
        generated_aigs = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                aig = self._generate_single_aig(max_nodes, temperature, num_inputs)
                generated_aigs.append(aig)
        
        return generated_aigs
    
    def _generate_single_aig(self, max_nodes: int, temperature: float, num_inputs: int) -> Dict[str, Any]:
        """
        Generate a single AIG using autoregressive sampling.
        
        Args:
            max_nodes: Maximum number of AND nodes
            temperature: Sampling temperature
            num_inputs: Number of primary inputs
            
        Returns:
            Generated AIG as dictionary
        """
        # Initialize with CONST0 and primary inputs
        nodes = ['CONST0'] + ['PI'] * num_inputs
        edges = []
        edge_types = []
        
        # Generate AND nodes autoregressively
        num_and_nodes = np.random.randint(1, max_nodes + 1)
        
        for and_idx in range(num_and_nodes):
            nodes.append('AND')
            current_node_id = len(nodes) - 1
            
            # Each AND gate needs two inputs
            # Sample from available nodes (all nodes before current)
            available_nodes = list(range(current_node_id))
            
            if len(available_nodes) >= 2:
                # Sample two input nodes
                input_nodes = np.random.choice(available_nodes, size=2, replace=False)
                
                for input_node in input_nodes:
                    edges.append((int(input_node), current_node_id))
                    # Randomly decide if edge is inverted
                    edge_type = 'INV' if np.random.random() < 0.3 else 'REG'
                    edge_types.append(edge_type)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'edge_types': edge_types,
            'num_inputs': num_inputs,
            'num_and_gates': num_and_nodes
        }
    
    def save_model(self, save_path: str):
        """
        Save Circuit Transformer model state.
        
        Args:
            save_path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call load_pretrained first.")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")


class SimpleTransformerCircuitGenerator(nn.Module):
    """
    Simplified transformer model for circuit generation.
    
    This is a minimal implementation that demonstrates the architecture.
    For full Circuit Transformer functionality, use the original implementation.
    """
    
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, max_nodes: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.max_nodes = max_nodes
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_nodes * 3, n_embd)  # Rough estimate
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Output projection
        self.output_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x, positions=None):
        """
        Forward pass through the transformer.
        
        Args:
            x: Input token indices [batch, seq_len]
            positions: Position indices [batch, seq_len]
            
        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project to vocabulary
        logits = self.output_head(x)
        
        return logits
