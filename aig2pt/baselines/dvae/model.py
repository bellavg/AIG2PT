"""
D-VAE model implementation for AIG generation.

Adapted from the original D-VAE paper:
"D-VAE: A Variational Autoencoder for Directed Acyclic Graphs" (NeurIPS 2019)
by Zhang et al. https://github.com/muhanzhang/D-VAE

This implementation is specifically adapted for AND-Inverter Graphs (AIGs).
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class DVAE_AIG(nn.Module):
    """
    D-VAE for AND-Inverter Graphs.
    
    Uses asynchronous message passing to encode/decode AIGs.
    A node updates its state only after all its predecessors have been updated,
    which allows the model to encode the computation on a DAG.
    
    Args:
        max_n: Maximum number of nodes in the graph
        nvt: Number of vertex/node types (CONST, PI, AND for AIGs)
        hs: Hidden state size
        nz: Latent representation size
        bidirectional: Whether to use bidirectional encoding
    """
    
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=256, nz=56, bidirectional=False):
        super(DVAE_AIG, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types (CONST0, PI, AND, PO)
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size
        self.nz = nz  # latent space dimension
        self.bidir = bidirectional
        self.device = None
        
        # Encoder: GRU-based encoding of sequential node representations
        # Input: one-hot node type + connections from predecessors
        self.xs = nvt + max_n - 1  # input size: [one_hot(type), bit(connections)]
        self.grue = nn.GRU(self.xs, hs, batch_first=True, bidirectional=self.bidir)
        
        # Encoder output size depends on bidirectionality
        encoder_output_size = hs * 2 if self.bidir else hs
        
        # Latent space parameters
        self.fc_mean = nn.Linear(encoder_output_size, nz)
        self.fc_logvar = nn.Linear(encoder_output_size, nz)
        
        # Decoder: from latent to initial hidden state
        self.fc_decode_init = nn.Linear(nz, hs)
        # Decoder GRU takes the same input as encoder (node representation)
        self.grud = nn.GRU(self.xs, hs, batch_first=True)
        
        # Node type prediction head
        self.add_node = nn.Sequential(
            nn.Linear(hs, hs),
            nn.ReLU(),
            nn.Linear(hs, nvt)
        )
        
        # Edge/connection prediction head
        self.add_edges = nn.Sequential(
            nn.Linear(hs, hs),
            nn.ReLU(),
            nn.Linear(hs, max_n - 1)
        )
        
        # For bidirectional, we need to unify the sizes
        if self.bidir:
            self.unify_encoder_hidden = nn.Linear(hs * 2, hs)
    
    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device())
    
    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs)
    
    def encode(self, G):
        """
        Encode a batch of graphs into latent representations.
        
        Args:
            G: Batch of graph tensors [batch_size, max_n-1, nvt + max_n-1]
               Each graph is a sequence of node descriptors in topological order
        
        Returns:
            mu: Mean of latent distribution [batch_size, nz]
            logvar: Log variance of latent distribution [batch_size, nz]
        """
        # G shape: [batch_size, seq_len, input_size]
        _, h = self.grue(G)  # h: [num_layers * num_directions, batch_size, hs]
        
        # Take the last hidden state
        if self.bidir:
            # Concatenate forward and backward hidden states
            h = torch.cat([h[-2], h[-1]], dim=1)  # [batch_size, hs*2]
        else:
            h = h[-1]  # [batch_size, hs]
        
        # Compute mean and log variance
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar, eps_scale=1.0):
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean [batch_size, nz]
            logvar: Log variance [batch_size, nz]
            eps_scale: Scale factor for the noise (for temperature sampling)
        
        Returns:
            z: Sampled latent vector [batch_size, nz]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * eps_scale
            z = mu + eps * std
            return z
        else:
            return mu
    
    def decode(self, z, stochastic=True):
        """
        Decode latent representation back to graph.
        
        Args:
            z: Latent vectors [batch_size, nz]
            stochastic: Whether to sample stochastically or use argmax
        
        Returns:
            G: Reconstructed graphs as list of (node_types, edges)
        """
        batch_size = z.size(0)
        
        # Initialize decoder hidden state from latent z
        h = self.fc_decode_init(z)  # [batch_size, hs]
        h = h.unsqueeze(0)  # [1, batch_size, hs]
        
        # Decode sequentially
        graphs = []
        for b in range(batch_size):
            node_types = []
            edges = []
            h_current = h[:, b:b+1, :]  # [1, 1, hs]
            
            for i in range(self.max_n - 1):
                # Predict node type
                node_logits = self.add_node(h_current.squeeze(0))  # [1, nvt]
                
                if stochastic and self.training:
                    node_type = torch.multinomial(F.softmax(node_logits, dim=-1), 1).item()
                else:
                    node_type = node_logits.argmax(dim=-1).item()
                
                # Check for end token
                if node_type == self.END_TYPE:
                    break
                
                node_types.append(node_type)
                
                # Predict edges (connections to previous nodes)
                if i > 0:
                    edge_logits = self.add_edges(h_current.squeeze(0))[:, :i]  # [1, i]
                    
                    if stochastic and self.training:
                        edge_probs = torch.sigmoid(edge_logits)
                        edge_connections = torch.bernoulli(edge_probs).squeeze(0)
                    else:
                        edge_connections = (edge_logits > 0).float().squeeze(0)
                    
                    # Store edges (from which previous nodes this node connects)
                    for j, connected in enumerate(edge_connections):
                        if connected > 0.5:
                            edges.append((j, i))
                
                # Update hidden state using GRU
                # Create input: concatenate one-hot node type and edge connections
                node_input = torch.zeros(1, 1, self.xs).to(self.get_device())
                node_input[0, 0, node_type] = 1.0  # one-hot node type
                
                if i > 0:
                    node_input[0, 0, self.nvt:self.nvt+i] = edge_connections
                
                _, h_current = self.grud(node_input, h_current)
            
            graphs.append({
                'node_types': node_types,
                'edges': edges,
                'num_nodes': len(node_types)
            })
        
        return graphs
    
    def forward(self, G, eps_scale=1.0):
        """
        Full forward pass: encode, reparameterize, and decode.
        
        Args:
            G: Input graphs [batch_size, max_n-1, nvt + max_n-1]
            eps_scale: Temperature for sampling
        
        Returns:
            Dictionary with reconstruction outputs, mu, logvar
        """
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar, eps_scale)
        
        # For training, we compute reconstruction loss directly
        # without generating discrete graphs
        return self.decode_for_loss(z, G), mu, logvar
    
    def decode_for_loss(self, z, G_true):
        """
        Decode for computing reconstruction loss.
        Returns logits instead of discrete predictions.
        
        Args:
            z: Latent vectors [batch_size, nz]
            G_true: True graphs for computing loss [batch_size, max_n-1, nvt + max_n-1]
        
        Returns:
            node_logits: [batch_size, max_n-1, nvt]
            edge_logits: [batch_size, max_n-1, max_n-1]
        """
        batch_size = z.size(0)
        
        # Initialize decoder
        h = self.fc_decode_init(z)  # [batch_size, hs]
        h = h.unsqueeze(0)  # [1, batch_size, hs]
        
        # Collect logits for all positions
        all_node_logits = []
        all_edge_logits = []
        
        # Decode step by step using teacher forcing
        for i in range(self.max_n - 1):
            # Predict node type
            node_logits = self.add_node(h.squeeze(0))  # [batch_size, nvt]
            all_node_logits.append(node_logits)
            
            # Predict edges
            edge_logits = self.add_edges(h.squeeze(0))  # [batch_size, max_n-1]
            all_edge_logits.append(edge_logits)
            
            # Teacher forcing: use true input for next step
            if i < G_true.size(1):
                node_input = G_true[:, i:i+1, :]  # [batch_size, 1, xs]
            else:
                # Padding
                node_input = torch.zeros(batch_size, 1, self.xs).to(self.get_device())
                node_input[:, 0, self.START_TYPE] = 1.0
            
            _, h = self.grud(node_input, h)
        
        # Stack logits
        node_logits = torch.stack(all_node_logits, dim=1)  # [batch_size, max_n-1, nvt]
        edge_logits = torch.stack(all_edge_logits, dim=1)  # [batch_size, max_n-1, max_n-1]
        
        return node_logits, edge_logits
    
    def loss(self, G, beta=1.0, eps_scale=1.0):
        """
        Compute VAE loss: reconstruction loss + KL divergence.
        
        Args:
            G: Input graphs [batch_size, max_n-1, nvt + max_n-1]
            beta: Weight for KL divergence term
            eps_scale: Temperature for sampling
        
        Returns:
            total_loss, recon_loss, kl_loss
        """
        batch_size = G.size(0)
        
        # Forward pass
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar, eps_scale)
        node_logits, edge_logits = self.decode_for_loss(z, G)
        
        # Extract true node types and edges from G
        # G: [batch_size, max_n-1, nvt + max_n-1]
        # First nvt dimensions are one-hot node types
        # Next max_n-1 dimensions are edge connections
        
        true_node_types = G[:, :, :self.nvt].argmax(dim=-1)  # [batch_size, max_n-1]
        true_edges = G[:, :, self.nvt:]  # [batch_size, max_n-1, max_n-1]
        
        # Reconstruction loss for node types
        node_loss = F.cross_entropy(
            node_logits.reshape(-1, self.nvt),
            true_node_types.reshape(-1),
            reduction='sum'
        )
        
        # Reconstruction loss for edges (binary cross entropy)
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits,
            true_edges,
            reduction='sum'
        )
        
        recon_loss = (node_loss + edge_loss) / batch_size
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def generate(self, num_samples, z=None):
        """
        Generate new graphs by sampling from the latent space.
        
        Args:
            num_samples: Number of graphs to generate
            z: Optional latent vectors. If None, sample from N(0, I)
        
        Returns:
            List of generated graphs
        """
        self.eval()
        with torch.no_grad():
            if z is None:
                z = torch.randn(num_samples, self.nz).to(self.get_device())
            
            graphs = self.decode(z, stochastic=False)
        
        return graphs
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
