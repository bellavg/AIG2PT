"""
Sampling script for LayerDAG on AIG datasets.

This script loads a trained LayerDAG model and generates synthetic AIGs
for unconditional generation evaluation.
"""
import os
import argparse
import torch
from tqdm import tqdm

from setup_utils import set_seed
from dataset import AIGDAGDataset


def sample_aigs(args, device, model, num_samples):
    """
    Sample AIGs using a trained LayerDAG model.
    
    Note: This is a placeholder for the full sampling implementation.
    Full LayerDAG sampling requires node diffusion and edge diffusion models.
    """
    syn_set = AIGDAGDataset(num_categories=3, label=False)
    
    print(f"Sampling {num_samples} AIGs...")
    print("Note: Full LayerDAG sampling implementation requires:")
    print("  1. Trained node count prediction model")
    print("  2. Trained node type diffusion model")
    print("  3. Trained edge diffusion model")
    print("\nCurrent implementation loads node count model only.")
    print("For complete sampling, train all three stages and use LayerDAG.sample().")
    
    return syn_set


def main(args):
    """Main sampling function."""
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location=device)
    
    # Load model
    from model.layer_dag import BiMPNNEncoder
    num_x_n_cat = torch.tensor([ckpt['num_categories']])
    
    model = BiMPNNEncoder(
        num_x_n_cat=num_x_n_cat,
        pe_emb_size=0,
        y_emb_size=0,
        pe='relative_level',
        **ckpt['config']['model']
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Sample AIGs
    syn_set = sample_aigs(args, device, model, args.num_samples)
    
    # Save samples
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'sampled_aigs.pth')
    
    # Convert to dictionary format
    data_dict = {
        'src_list': [],
        'dst_list': [],
        'x_n_list': [],
    }
    
    for i in range(len(syn_set)):
        if syn_set.conditional:
            src_i, dst_i, x_n_i, y_i = syn_set[i]
        else:
            src_i, dst_i, x_n_i = syn_set[i]
        
        data_dict['src_list'].append(src_i)
        data_dict['dst_list'].append(dst_i)
        data_dict['x_n_list'].append(x_n_i)
    
    torch.save(data_dict, output_path)
    print(f"Saved {len(syn_set)} sampled AIGs to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample AIGs using trained LayerDAG model'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of AIGs to sample'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./samples',
        help='Directory to save sampled AIGs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    
    args = parser.parse_args()
    main(args)
