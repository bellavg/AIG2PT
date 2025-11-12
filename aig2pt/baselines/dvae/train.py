"""
Training script for D-VAE baseline on AIG generation.
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from .model import DVAE_AIG
from .data_loader import load_aig_data


def train_epoch(model, train_loader, optimizer, device, beta=1.0, eps_scale=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Compute loss
        loss, recon_loss, kl_loss = model.loss(batch, beta=beta, eps_scale=eps_scale)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches
    }


def validate(model, val_loader, device, beta=1.0):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(device)
            
            # Compute loss
            loss, recon_loss, kl_loss = model.loss(batch, beta=beta, eps_scale=1.0)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'kl': total_kl / num_batches
    }


def train_dvae(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, _ = load_aig_data(
        data_dir=args.data_dir,
        max_n=args.max_n,
        nvt=args.nvt,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if train_loader is None:
        raise ValueError("No training data found!")
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = DVAE_AIG(
        max_n=args.max_n,
        nvt=args.nvt,
        START_TYPE=0,
        END_TYPE=1,
        hs=args.hidden_size,
        nz=args.latent_size,
        bidirectional=args.bidirectional
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            beta=args.beta, eps_scale=args.eps_scale
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon']:.4f}, "
              f"KL: {train_metrics['kl']:.4f}")
        
        # Validate
        if val_loader:
            val_metrics = validate(model, val_loader, device, beta=args.beta)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon']:.4f}, "
                  f"KL: {val_metrics['kl']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, output_dir / 'best_model.pt')
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoint periodically
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'final_model.pt')
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Train D-VAE for AIG generation')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='./dvae_output',
                        help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--max-n', type=int, default=56,
                        help='Maximum number of nodes')
    parser.add_argument('--nvt', type=int, default=6,
                        help='Number of vertex types (including START/END)')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size for GRU')
    parser.add_argument('--latent-size', type=int, default=56,
                        help='Latent space dimension')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional encoding')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for KL divergence term')
    parser.add_argument('--eps-scale', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for learning rate scheduling')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    train_dvae(args)


if __name__ == '__main__':
    main()
