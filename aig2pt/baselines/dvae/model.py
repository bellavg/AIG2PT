"""
D-VAE baseline adapter for AIG generation.

This module provides an adapter for the D-VAE (Variational Autoencoder
for Directed Acyclic Graphs) model to work with the AIG2PT data format
for unconditional generation.

D-VAE uses asynchronous message passing in the encoder and a
sequential decoder to generate DAGs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add parent directory to path for imports
BASELINES_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASELINES_DIR))

import base_model
BaselineModel = base_model.BaselineModel


class DVAEBaseline(BaselineModel):
    """
    Adapter for D-VAE model for unconditional AIG generation.
    
    D-VAE uses a variational autoencoder architecture specifically designed
    for directed acyclic graphs with asynchronous message passing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize D-VAE baseline.
        
        Args:
            config: Configuration dictionary with keys:
                - hidden_dim: Hidden dimension for encoder/decoder
                - latent_dim: Dimension of latent space
                - max_nodes: Maximum number of nodes
                - num_node_types: Number of node types (e.g., 3 for CONST0, PI, AND)
        """
        super().__init__(config)
        
        # Model hyperparameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.latent_dim = config.get('latent_dim', 64)
        self.max_nodes = config.get('max_nodes', 50)
        self.num_node_types = config.get('num_node_types', 3)
        
        # Node and edge type mappings for AIG
        self.node_types = ['CONST0', 'PI', 'AND']
        self.edge_types = ['REG', 'INV']
        
    def load_pretrained(self, checkpoint_path: Optional[str] = None):
        """
        Load pretrained D-VAE model or initialize from scratch.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, initialize randomly.
        """
        # Build D-VAE model
        self.model = DVAE(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            max_nodes=self.max_nodes,
            num_node_types=self.num_node_types
        ).to(self.device)
        
        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded D-VAE checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Initializing with random weights")
        else:
            print("Initializing D-VAE with random weights")
    
    def generate(self, 
                 num_samples: int = 1,
                 max_nodes: int = 50,
                 temperature: float = 1.0,
                 num_inputs: int = 4,
                 **kwargs) -> List[Dict[str, Any]]:
        """
        Generate AIGs unconditionally using D-VAE.
        
        Args:
            num_samples: Number of AIGs to generate
            max_nodes: Maximum number of AND nodes
            temperature: Sampling temperature (used for latent sampling)
            num_inputs: Number of primary inputs
            **kwargs: Additional parameters
            
        Returns:
            List of generated AIGs in standard format
        """
        self.model.eval()
        generated_aigs = []
        
        with torch.no_grad():
            # Sample from prior (standard normal)
            z = torch.randn(num_samples, self.latent_dim).to(self.device) * temperature
            
            # Decode to generate AIGs
            for i in range(num_samples):
                aig = self._decode_to_aig(z[i:i+1], max_nodes, num_inputs)
                generated_aigs.append(aig)
        
        return generated_aigs
    
    def _decode_to_aig(self, z: torch.Tensor, max_nodes: int, num_inputs: int) -> Dict[str, Any]:
        """
        Decode latent vector to AIG.
        
        Args:
            z: Latent vector [1, latent_dim]
            max_nodes: Maximum number of AND nodes
            num_inputs: Number of primary inputs
            
        Returns:
            Generated AIG as dictionary
        """
        # Decode using the D-VAE decoder
        node_logits, edge_probs = self.model.decode(z, max_nodes + num_inputs + 1)
        
        # Initialize with CONST0 and primary inputs
        nodes = ['CONST0'] + ['PI'] * num_inputs
        edges = []
        edge_types = []
        
        # Sample AND nodes
        num_and_nodes = np.random.randint(1, max_nodes + 1)
        
        # Generate AND gates
        for and_idx in range(num_and_nodes):
            nodes.append('AND')
            current_node_id = len(nodes) - 1
            
            # Sample edges based on probabilities
            available_nodes = list(range(current_node_id))
            
            if len(available_nodes) >= 2:
                # Get edge probabilities for this node
                if current_node_id < edge_probs.shape[1]:
                    probs = edge_probs[0, current_node_id, :current_node_id].cpu().numpy()
                    
                    # Select top-2 connections
                    top_indices = np.argsort(probs)[-2:]
                    
                    for input_node in top_indices:
                        edges.append((int(input_node), current_node_id))
                        # Randomly decide if edge is inverted
                        edge_type = 'INV' if np.random.random() < 0.3 else 'REG'
                        edge_types.append(edge_type)
                else:
                    # Fallback: random connections
                    input_nodes = np.random.choice(available_nodes, size=2, replace=False)
                    for input_node in input_nodes:
                        edges.append((int(input_node), current_node_id))
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
        Save D-VAE model state.
        
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


class DVAE(nn.Module):
    """
    Simplified D-VAE (Variational Autoencoder for DAGs).
    
    This is a minimal implementation demonstrating the VAE approach for DAGs.
    For full D-VAE functionality, use the original implementation.
    """
    
    def __init__(self, hidden_dim: int, latent_dim: int, max_nodes: int, num_node_types: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.num_node_types = num_node_types
        
        # Encoder: DAG -> latent
        self.encoder = DAGEncoder(hidden_dim, latent_dim, num_node_types)
        
        # Decoder: latent -> DAG
        self.decoder = DAGDecoder(hidden_dim, latent_dim, max_nodes, num_node_types)
    
    def encode(self, node_features, adj_matrix):
        """
        Encode DAG to latent distribution.
        
        Args:
            node_features: Node feature matrix [batch, num_nodes, feature_dim]
            adj_matrix: Adjacency matrix [batch, num_nodes, num_nodes]
            
        Returns:
            mu, logvar: Mean and log-variance of latent distribution
        """
        return self.encoder(node_features, adj_matrix)
    
    def decode(self, z, num_nodes):
        """
        Decode latent vector to DAG.
        
        Args:
            z: Latent vector [batch, latent_dim]
            num_nodes: Number of nodes to generate
            
        Returns:
            node_logits, edge_probs: Node type logits and edge probabilities
        """
        return self.decoder(z, num_nodes)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean [batch, latent_dim]
            logvar: Log-variance [batch, latent_dim]
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, node_features, adj_matrix):
        """
        Full forward pass through VAE.
        
        Args:
            node_features: Node features
            adj_matrix: Adjacency matrix
            
        Returns:
            Reconstruction and latent parameters
        """
        mu, logvar = self.encode(node_features, adj_matrix)
        z = self.reparameterize(mu, logvar)
        node_logits, edge_probs = self.decode(z, node_features.shape[1])
        return node_logits, edge_probs, mu, logvar


class DAGEncoder(nn.Module):
    """Encoder network for D-VAE."""
    
    def __init__(self, hidden_dim: int, latent_dim: int, num_node_types: int):
        super().__init__()
        self.node_embedding = nn.Embedding(num_node_types, hidden_dim)
        self.gnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, node_features, adj_matrix):
        """Encode graph to latent distribution."""
        # Simple graph pooling
        h = self.gnn(node_features)
        h_pooled = torch.mean(h, dim=1)  # Global mean pooling
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        return mu, logvar


class DAGDecoder(nn.Module):
    """Decoder network for D-VAE."""
    
    def __init__(self, hidden_dim: int, latent_dim: int, max_nodes: int, num_node_types: int):
        super().__init__()
        self.max_nodes = max_nodes
        self.fc_z = nn.Linear(latent_dim, hidden_dim)
        self.node_decoder = nn.Linear(hidden_dim, num_node_types)
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z, num_nodes):
        """Decode latent to graph structure."""
        h = F.relu(self.fc_z(z))
        
        # Expand for each node
        h_expanded = h.unsqueeze(1).expand(-1, num_nodes, -1)
        
        # Decode node types
        node_logits = self.node_decoder(h_expanded)
        
        # Decode edges
        batch_size = z.shape[0]
        edge_probs = torch.zeros(batch_size, num_nodes, num_nodes).to(z.device)
        
        for i in range(num_nodes):
            for j in range(i):  # Only lower triangular (DAG property)
                h_pair = torch.cat([h_expanded[:, i], h_expanded[:, j]], dim=-1)
                edge_probs[:, i, j] = self.edge_decoder(h_pair).squeeze(-1)
        
        return node_logits, edge_probs
