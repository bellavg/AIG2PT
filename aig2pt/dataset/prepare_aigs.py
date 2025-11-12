# AIG2PT/dataset/prepare_aigs.py
# PURPOSE: Convert AIGER files to tokenized text sequences for transformer training.
# APPROACH: Similar to G2PT's molecule pipeline - convert to text, tokenize, save token IDs.
#
# Process:
# 1. Parse .aig files with aigverse
# 2. Convert to text sequences with topological ordering (nodes + edges)
# 3. Add synthetic PO nodes (like aigverse networkx adapter)
# 4. Tokenize using pre-built tokenizer
# 5. Save as token ID arrays (.bin files) with memmap for efficiency

import os
import sys
import math
import numpy as np
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import warnings

# --- Installation Check ---
try:
    from aigverse import read_aiger_into_aig, read_ascii_aiger_into_aig, to_edge_list
except ImportError:
    print("Error: 'aigverse' library not found.")
    print("Please install it by running: pip install aigverse")
    sys.exit(1)

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Load config to get paths
import yaml
CONFIG_PATH = SCRIPT_DIR.parent / 'configs' / 'aig.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Paths from config
raw_data_dir = config.get('raw_data_dir')
if not raw_data_dir:
    print("Error: 'raw_data_dir' missing from config.")
    sys.exit(1)

raw_data_path = Path(raw_data_dir)
if not raw_data_path.is_absolute():
    raw_data_path = (CONFIG_PATH.parent / raw_data_dir).resolve()
INPUT_AIGER_DIR = raw_data_path

TOKENIZER_PATH = SCRIPT_DIR / 'tokenizer'

processed_data_dir = config.get('processed_data_dir')
if processed_data_dir:
    processed_path = Path(processed_data_dir)
    if not processed_path.is_absolute():
        processed_path = (CONFIG_PATH.parent / processed_data_dir).resolve()
else:
    processed_path = SCRIPT_DIR / 'aig_prepared'
FINAL_OUTPUT_DIR = processed_path

# Train/validation/test split ratios from config
try:
    VAL_SPLIT_RATIO = float(config.get('val_split_ratio', 0.1))
    TEST_SPLIT_RATIO = float(config.get('test_split_ratio', 0.1))
except (TypeError, ValueError):
    print("Error: 'val_split_ratio' and 'test_split_ratio' must be numeric values.")
    sys.exit(1)

if VAL_SPLIT_RATIO < 0 or TEST_SPLIT_RATIO < 0:
    print("Error: split ratios cannot be negative.")
    sys.exit(1)

TRAIN_SPLIT_RATIO = 1.0 - VAL_SPLIT_RATIO - TEST_SPLIT_RATIO
if TRAIN_SPLIT_RATIO <= 0:
    print("Error: Train split ratio must be positive. Please adjust validation/test ratios in the config.")
    sys.exit(1)

# PAD_TOKEN_ID will be fetched from tokenizer during runtime

# ==============================================================================


def aig_to_text_sequence(aig, filepath: str, root_dir: str):
    """
    Converts an AIG to a text sequence following G2PT format.

    Format: <boc> [NODE_SECTION] <eoc> <bog> [EDGE_SECTION] <eog> <eos>

    Node section: <sepc> NODE_TYPE IDX_i for each node
    Edge section: <sepg> IDX_src IDX_dst EDGE_TYPE for each edge (in topological order)

    Includes synthetic PO nodes (like aigverse networkx adapter).

    Args:
        aig: aigverse.Aig object
        filepath: Source file path
        root_dir: Root data directory for creating relative path ID

    Returns:
        tuple: (text_sequence: str, graph_id: str)
    """
    # Extract graph ID as relative path from root data folder
    # e.g., "10/6/adder_c_10_l_6_w_0.aig"
    graph_id = str(Path(filepath).relative_to(Path(root_dir)))

    # Get basic components
    pi_nodes = aig.pis()        # List of PI nodes
    po_signals = aig.pos()      # List of PO signals
    gate_nodes = aig.gates()    # List of AND gate nodes
    num_nodes = aig.size()

    # Build node section following topological order from aigverse
    node_tokens = ["<boc>"]

    # Node 0: constant (always present)
    node_tokens.extend(["<sepc>", "NODE_CONST0", "IDX_0"])

    # Primary inputs
    for pi_node in pi_nodes:
        node_idx = aig.node_to_index(pi_node)
        node_tokens.extend(["<sepc>", "NODE_PI", f"IDX_{node_idx}"])

    # AND gates (already in topological order from aigverse)
    for gate_node in gate_nodes:
        node_idx = aig.node_to_index(gate_node)
        node_tokens.extend(["<sepc>", "NODE_AND", f"IDX_{node_idx}"])

    # Synthetic PO nodes (like aigverse networkx adapter)
    # PO nodes have indices: num_nodes, num_nodes+1, ...
    for po_idx in range(len(po_signals)):
        synthetic_po_index = num_nodes + po_idx
        node_tokens.extend(["<sepc>", "NODE_PO", f"IDX_{synthetic_po_index}"])

    node_tokens.append("<eoc>")

    # Build edge section in topological order
    # to_edge_list() already includes ALL edges, including edges to synthetic PO nodes
    edge_tokens = ["<bog>"]

    edges = to_edge_list(aig)
    for edge in edges:
        # edge.source and edge.target are already integers (node indices)
        src_idx = edge.source
        tgt_idx = edge.target
        edge_type = "EDGE_INV" if edge.weight else "EDGE_REG"

        edge_tokens.extend(["<sepg>", f"IDX_{src_idx}", f"IDX_{tgt_idx}", edge_type])

    edge_tokens.append("<eog>")

    # Combine all parts (NO <eos> token - matching original G2PT)
    all_tokens = node_tokens + edge_tokens
    text_sequence = " ".join(all_tokens)

    return text_sequence, graph_id


def process_aiger_file(filepath: str, tokenizer, root_dir: str):
    """
    Loads an AIGER file, converts to text sequence, and tokenizes.

    Args:
        filepath: Path to .aig file
        tokenizer: HuggingFace tokenizer
        root_dir: Root data directory for creating relative path IDs

    Returns:
        tuple: (token_ids: torch.Tensor, graph_id: str) or None if error
    """
    try:
        # Load AIG using correct aigverse API
        filepath_obj = Path(filepath)
        if filepath_obj.suffix == '.aag':
            aig = read_ascii_aiger_into_aig(str(filepath))
        else:  # .aig (binary format)
            aig = read_aiger_into_aig(str(filepath))

        # Convert to text sequence with root-relative path as ID
        text_sequence, graph_id = aig_to_text_sequence(aig, filepath, root_dir)

        # Tokenize
        encoded = tokenizer(
            text_sequence,
            padding=False,  # We'll pad in batches later
            truncation=False,  # Don't truncate
            return_tensors='pt'
        )

        token_ids = encoded['input_ids'].squeeze(0)  # Remove batch dimension

        return token_ids, graph_id

    except Exception as e:
        warnings.warn(f"Skipping file {Path(filepath).name} due to error: {e}")
        return None


if __name__ == '__main__':
    print(f"--- AIG to Tokenized Sequence Conversion ---")
    print(f"Input AIGER Directory: {INPUT_AIGER_DIR}")
    print(f"Tokenizer Path: {TOKENIZER_PATH}")
    print(f"Final Output Directory: {FINAL_OUTPUT_DIR}")

    # --- Validate Paths ---
    input_path = Path(INPUT_AIGER_DIR)
    if not input_path.is_dir():
        print(f"Error: Input directory not found at '{INPUT_AIGER_DIR}'")
        sys.exit(1)

    if not TOKENIZER_PATH.is_dir():
        print(f"Error: Tokenizer not found at '{TOKENIZER_PATH}'")
        print("Please run setup.py first to generate the tokenizer.")
        sys.exit(1)

    # --- Load Tokenizer ---
    print(f"\nLoading tokenizer from {TOKENIZER_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        print(f"  ✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {len(tokenizer)}")

        # Get PAD token ID from tokenizer
        PAD_TOKEN_ID = tokenizer.pad_token_id
        if PAD_TOKEN_ID is None:
            print(f"  Warning: Tokenizer has no pad_token_id, using 0")
            PAD_TOKEN_ID = 0
        else:
            print(f"  PAD token ID: {PAD_TOKEN_ID}")

    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # --- Collect AIG Files ---
    print(f"\nScanning for .aig and .aag files...")
    all_aiger_files = list(input_path.glob('**/*.aig')) + list(input_path.glob('**/*.aag'))

    if not all_aiger_files:
        print(f"Error: No AIGER files (.aig, .aag) found in '{INPUT_AIGER_DIR}'")
        sys.exit(1)

    print(f"Found {len(all_aiger_files)} AIGER files.")

    # --- Shuffle and Split into Train/Val/Test ---
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(all_aiger_files))

    # Calculate split indices
    n_total = len(all_aiger_files)
    n_test = int(n_total * TEST_SPLIT_RATIO)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_train = n_total - n_test - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_files = [all_aiger_files[i] for i in train_indices]
    val_files = [all_aiger_files[i] for i in val_indices]
    test_files = [all_aiger_files[i] for i in test_indices]

    file_splits = {'train': train_files, 'val': val_files, 'test': test_files}
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # --- Process Each Split ---
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {}
    overall_max_sequence_length = 0

    for split_name, files in file_splits.items():
        if not files:
            print(f"\nSkipping split '{split_name}': No files.")
            continue

        print(f"\n--- Processing split: {split_name} ({len(files)} files) ---")
        split_output_dir = FINAL_OUTPUT_DIR / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Lists to store data
        token_ids_list = []
        graph_ids_list = []
        num_processed = 0

        for filepath in tqdm(files, desc=f"  Converting {split_name}"):
            result = process_aiger_file(str(filepath), tokenizer, str(INPUT_AIGER_DIR))
            if result:
                token_ids, graph_id = result
                token_ids_list.append(token_ids)
                graph_ids_list.append(graph_id)
                num_processed += 1

        if num_processed == 0:
            print(f"  Warning: No valid graphs processed for '{split_name}'")
            continue

        print(f"  Successfully processed {num_processed}/{len(files)} files")

        # --- Pad Sequences ---
        print(f"  Padding sequences to max length...")
        token_ids_padded = pad_sequence(
            token_ids_list,
            batch_first=True,
            padding_value=PAD_TOKEN_ID
        )
        token_ids_np = token_ids_padded.numpy().astype(np.int16)

        print(f"    Shape: {token_ids_np.shape}")
        print(f"    Max sequence length: {token_ids_np.shape[1]}")

        # Track global max sequence length
        overall_max_sequence_length = max(overall_max_sequence_length, int(token_ids_np.shape[1]))

        # --- Save with Memmap ---
        print(f"  Saving to {split_output_dir}...")

        # Save token IDs
        token_ids_path = split_output_dir / 'token_ids.bin'
        memmap_tokens = np.memmap(
            token_ids_path,
            dtype=np.int16,
            mode='w+',
            shape=token_ids_np.shape
        )
        memmap_tokens[:] = token_ids_np
        memmap_tokens.flush()
        del memmap_tokens

        # Save graph IDs as JSON
        graph_ids_path = split_output_dir / 'graph_ids.json'
        with open(graph_ids_path, 'w', encoding='utf-8') as f:
            json.dump(graph_ids_list, f, indent=2)

        # Store metadata
        metadata[split_name] = {
            'num_graphs': num_processed,
            'token_ids_shape': list(token_ids_np.shape),
            'max_sequence_length': int(token_ids_np.shape[1]),
            'files': {
                'token_ids': str(token_ids_path.relative_to(FINAL_OUTPUT_DIR)),
                'graph_ids': str(graph_ids_path.relative_to(FINAL_OUTPUT_DIR))
            }
        }

        print(f"  ✓ Saved {split_name} data")

    # --- Save Metadata ---
    if overall_max_sequence_length > 0:
        metadata['overall'] = {
            'max_sequence_length': overall_max_sequence_length
        }

    # Update block size and tokenizer config based on observed sequence lengths
    recommended_block_size = None
    if overall_max_sequence_length > 0:
        recommended_block_size = 1 << (overall_max_sequence_length - 1).bit_length()
        current_block_size = config.get('block_size', 1024)

        if recommended_block_size != current_block_size:
            print("\nUpdating block_size based on prepared data:")
            print(f"  Observed max sequence length: {overall_max_sequence_length}")
            print(f"  Recommended block_size (power of two): {recommended_block_size}")
            print(f"  Previous block_size: {current_block_size}")

            config['block_size'] = recommended_block_size
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

            tokenizer_config_path = TOKENIZER_PATH / 'tokenizer_config.json'
            try:
                with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                    tokenizer_config = json.load(f)
            except FileNotFoundError:
                print(f"Warning: tokenizer_config.json not found at '{tokenizer_config_path}'. Skipping update.")
            else:
                tokenizer_config['model_max_length'] = recommended_block_size
                with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
                    json.dump(tokenizer_config, f, indent=2)

        if recommended_block_size is not None:
            metadata.setdefault('overall', {})['recommended_block_size'] = recommended_block_size

    meta_path = FINAL_OUTPUT_DIR / 'data_meta.json'
    print(f"\nSaving metadata to: {meta_path}")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Conversion Complete! ---")
    print(f"Output directory: {FINAL_OUTPUT_DIR}")
    print(f"Ready for training!")
