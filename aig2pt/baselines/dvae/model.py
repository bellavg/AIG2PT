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
