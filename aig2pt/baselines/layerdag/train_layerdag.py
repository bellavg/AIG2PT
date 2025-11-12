"""
Training script for LayerDAG on AIG datasets.

This script trains a LayerDAG model for unconditional AIG generation.
The model learns to generate AIGs layer-by-layer using discrete diffusion.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from setup_utils import set_seed, load_yaml
from dataset import load_aig_dataset
from dataset.layer_dag import (
    LayerDAGNodeCountDataset,
    LayerDAGNodePredDataset,
    LayerDAGEdgePredDataset,
    collate_node_count,
    collate_node_pred,
    collate_edge_pred
)
from model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

try:
    import dgl.sparse as dglsp
except ImportError:
    print("Warning: DGL not installed. Please install with: pip install dgl")
    dglsp = None


@torch.no_grad()
def eval_node_count(device, val_loader, model):
    """Evaluate node count prediction."""
    if dglsp is None:
        raise ImportError("DGL is required for training")
    
    model.eval()
    total_nll = 0
    total_count = 0
    true_count = 0
    
    for batch_data in tqdm(val_loader, desc="Evaluating"):
        if len(batch_data) == 8:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                batch_rel_level, batch_y, batch_n2g_index, batch_label = batch_data
            batch_y = batch_y.to(device)
        else:
            batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                batch_rel_level, batch_n2g_index, batch_label = batch_data
            batch_y = None

        num_nodes = len(batch_x_n)
        batch_A = dglsp.spmatrix(
            batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
        batch_x_n = batch_x_n.to(device)
        batch_abs_level = batch_abs_level.to(device)
        batch_rel_level = batch_rel_level.to(device)
        batch_A_n2g = dglsp.spmatrix(
            batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_label = batch_label.to(device)

        batch_logits = model(batch_A, batch_x_n, batch_abs_level,
                             batch_rel_level, batch_A_n2g, batch_y)

        batch_nll = -batch_logits.log_softmax(dim=-1)
        batch_label = batch_label.clamp(max=batch_nll.shape[-1] - 1)
        batch_nll = batch_nll[torch.arange(batch_size).to(device), batch_label]
        total_nll += batch_nll.sum().item()

        batch_probs = batch_logits.softmax(dim=-1)
        batch_preds = batch_probs.multinomial(1).squeeze(-1)
        true_count += (batch_preds == batch_label).sum().item()

        total_count += batch_size

    avg_nll = total_nll / total_count if total_count > 0 else float('inf')
    accuracy = true_count / total_count if total_count > 0 else 0.0
    
    return avg_nll, accuracy


def train_node_count(device, train_set, val_set, model, config, patience):
    """Train the node count prediction model."""
    if dglsp is None:
        raise ImportError("DGL is required for training")
    
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        collate_fn=collate_node_count,
        **config['loader'],
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        collate_fn=collate_node_count,
        **config['loader']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    best_val_nll = float('inf')
    best_val_acc = 0
    best_state_dict = deepcopy(model.state_dict())
    num_patient_epochs = 0
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            if len(batch_data) == 8:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                    batch_rel_level, batch_y, batch_n2g_index, batch_label = batch_data
                batch_y = batch_y.to(device)
            else:
                batch_size, batch_edge_index, batch_x_n, batch_abs_level,\
                    batch_rel_level, batch_n2g_index, batch_label = batch_data
                batch_y = None

            num_nodes = len(batch_x_n)
            batch_A = dglsp.spmatrix(
                batch_edge_index, shape=(num_nodes, num_nodes)).to(device)
            batch_x_n = batch_x_n.to(device)
            batch_abs_level = batch_abs_level.to(device)
            batch_rel_level = batch_rel_level.to(device)
            batch_A_n2g = dglsp.spmatrix(
                batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_label = batch_label.to(device)

            batch_pred = model(batch_A, batch_x_n, batch_abs_level,
                               batch_rel_level, batch_A_n2g, batch_y)
            
            loss = criterion(batch_pred, batch_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        val_nll, val_acc = eval_node_count(device, val_loader, model)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val NLL={val_nll:.4f}, Val Acc={val_acc:.4f}")

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_acc = val_acc
            best_state_dict = deepcopy(model.state_dict())
            num_patient_epochs = 0
        else:
            num_patient_epochs += 1
            if num_patient_epochs >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    model.load_state_dict(best_state_dict)
    print(f"Best Val NLL: {best_val_nll:.4f}, Best Val Acc: {best_val_acc:.4f}")
    return model


def main(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_yaml(args.config_file)
    print("Configuration:")
    print(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading AIG dataset from {args.data_dir}...")
    train_set, val_set, test_set = load_aig_dataset(args.data_dir)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    if len(train_set) == 0:
        raise ValueError("Training set is empty!")
    
    # Get number of node categories from dataset
    num_x_n_cat = torch.tensor([train_set.num_categories])
    
    # Train stage 1: Node count prediction
    print("\n" + "="*50)
    print("Stage 1: Training Node Count Prediction")
    print("="*50)
    
    from model.layer_dag import BiMPNNEncoder
    node_count_model = BiMPNNEncoder(
        num_x_n_cat=num_x_n_cat,
        pe_emb_size=0,
        y_emb_size=0,  # Unconditional
        pe='relative_level',
        pool=config['node_count']['model']['pool'],
        **config['node_count']['model']
    ).to(device)
    
    # Create LayerDAG datasets for node count
    train_node_count = LayerDAGNodeCountDataset(train_set, conditional=False)
    val_node_count = LayerDAGNodeCountDataset(val_set, conditional=False)
    
    # Train node count model
    node_count_model = train_node_count(
        device, train_node_count, val_node_count, 
        node_count_model, config['node_count'], 
        config['general']['patience']
    )
    
    # Save node count model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    node_count_path = os.path.join(
        args.output_dir, f'node_count_model_{timestamp}.pth'
    )
    torch.save({
        'model_state_dict': node_count_model.state_dict(),
        'config': config['node_count'],
        'num_categories': train_set.num_categories
    }, node_count_path)
    print(f"Saved node count model to {node_count_path}")
    
    print("\nTraining complete!")
    print("Note: Full LayerDAG training includes node prediction and edge prediction stages.")
    print("This script currently implements node count prediction only.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train LayerDAG model on AIG dataset'
    )
    parser.add_argument(
        '--config_file', 
        type=str, 
        default='configs/aig.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to AIG dataset directory (containing raw/{train,val,test}.pt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    main(args)
