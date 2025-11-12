import os
import re
import yaml
import sys
import json
from pathlib import Path

# --- Configuration ---
# Get absolute path of this script's directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Path to the dataset-specific config file (will be read and updated)
DATASET_CONFIG_PATH = SCRIPT_DIR.parent / 'configs' / 'aig.yaml'
# Path to the base model config file (will be read only)
MODEL_CONFIG_PATH = SCRIPT_DIR.parent / 'configs' / 'network.yaml'
# Hardcoded path for saving the generated tokenizer files
TOKENIZER_OUTPUT_PATH = SCRIPT_DIR / 'tokenizer'


def parse_aiger_header(file_path):
    """
    Reads the first line of an AIGER file and parses the header.
    AIGER header format: aig M I L O A
    where M = max variable index, I = inputs, L = latches, O = outputs, A = AND gates
    """
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline()
            matches = re.findall(r'\d+', header_line)

            if header_line.startswith('aig') and len(matches) >= 5:
                pi_count = int(matches[1])
                po_count = int(matches[3])
                and_count = int(matches[4])

                # Calculate edge count for AIGs:
                # - Each AND gate has exactly 2 input edges
                # - Each PO (output) has 1 edge pointing to a node
                edge_count = (and_count * 2) + po_count

                return {
                    'max_node_count': int(matches[0]),
                    'pi_count': pi_count,
                    'po_count': po_count,
                    'and_count': and_count,
                    'edge_count': edge_count,
                }
            else:
                print(f"Warning: Skipping file '{file_path}' - invalid header.")
                return None
    except Exception as e:
        print(f"Error reading or parsing file {file_path}: {e}")
        return None


def analyze_aig_directory(directory_path):
    """
    Recursively analyzes a directory of AIGER files to find min/max stats.
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        print(f"Error: Data directory not found at '{directory_path}'")
        sys.exit(1)

    stats = {
        'max_node_count': 0,
        'pi_counts': {'min': float('inf'), 'max': 0},
        'po_counts': {'min': float('inf'), 'max': 0},
        'and_counts': {'min': float('inf'), 'max': 0},
        'edge_counts': {'min': float('inf'), 'max': 0},
    }

    file_count = 0
    print(f"Searching for .aag and .aig files in '{directory_path}'...")

    # Use glob for more efficient file searching
    aig_files = list(directory_path.glob('**/*.aig')) + list(directory_path.glob('**/*.aag'))
    total_files = len(aig_files)
    print(f"Found {total_files} AIGER files to analyze...")

    for file_path in aig_files:
        data = parse_aiger_header(file_path)
        if data:
            file_count += 1
            stats['max_node_count'] = max(stats['max_node_count'], data['max_node_count'])
            stats['pi_counts']['min'] = min(stats['pi_counts']['min'], data['pi_count'])
            stats['pi_counts']['max'] = max(stats['pi_counts']['max'], data['pi_count'])
            stats['po_counts']['min'] = min(stats['po_counts']['min'], data['po_count'])
            stats['po_counts']['max'] = max(stats['po_counts']['max'], data['po_count'])
            stats['and_counts']['min'] = min(stats['and_counts']['min'], data['and_count'])
            stats['and_counts']['max'] = max(stats['and_counts']['max'], data['and_count'])
            stats['edge_counts']['min'] = min(stats['edge_counts']['min'], data['edge_count'])
            stats['edge_counts']['max'] = max(stats['edge_counts']['max'], data['edge_count'])

        # Progress reporting for large datasets
        if file_count % 100 == 0:
            print(f"  Processed {file_count}/{total_files} files...")

    if file_count == 0:
        print(f"Warning: No valid AIGER files found in '{directory_path}'.")
        return None

    print(f"Analysis complete. Successfully processed {file_count}/{total_files} AIGER files.")

    # Replace inf with 0 for min values that were never set
    for key, value in stats.items():
        if isinstance(value, dict) and value['min'] == float('inf'):
            value['min'] = 0

    return stats


def run_analysis_and_update_config(config_path):
    """
    Reads config, analyzes AIGER files, and updates the config with stats.
    Returns the updated config data.
    """
    config_path = Path(config_path)
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        sys.exit(1)

    # Use 'raw_data_dir' for analysis, which should be in the config
    raw_aig_directory = config_data.get('raw_data_dir')
    if not raw_aig_directory:
        print(f"Error: 'raw_data_dir' not found in '{config_path}'. This is needed for analysis.")
        sys.exit(1)

    # Convert to Path and resolve relative to config file location
    raw_aig_path = Path(raw_aig_directory)
    if not raw_aig_path.is_absolute():
        # Resolve relative to the config file's parent directory
        raw_aig_path = (config_path.parent / raw_aig_directory).resolve()

    print(f"Analyzing AIG files in: {raw_aig_path}")

    final_stats = analyze_aig_directory(raw_aig_path)
    if not final_stats:
        return config_data  # Return original data if no stats were generated

    config_data.update({
        'max_node_count': final_stats['max_node_count'],
        'pi_counts': final_stats['pi_counts'],
        'po_counts': final_stats['po_counts'],
        'and_counts': final_stats['and_counts'],
        'edge_counts': final_stats['edge_counts']
    })

    # Calculate recommended block_size based on ACTUAL node and edge counts from data
    # Formula: (3 tokens per node) + (4 tokens per edge) + overhead
    # Node section: <boc> + (N nodes × 3 tokens each) + <eoc>
    # Edge section: <bog> + (E edges × 4 tokens each) + <eog>
    # Plus <eos> at the end
    max_nodes = final_stats['max_node_count']
    max_edges = final_stats['edge_counts']['max']

    # Actual calculation based on sequence format:
    # Node tokens: 1 (<boc>) + (max_nodes * 3) + 1 (<eoc>)
    # Edge tokens: 1 (<bog>) + (max_edges * 4) + 1 (<eog>)
    # Plus 1 for <eos>
    calculated_block_size = 1 + (max_nodes * 3) + 1 + 1 + (max_edges * 4) + 1 + 1

    # Round up to nearest power of 2 for efficiency
    import math
    recommended_block_size = 2 ** math.ceil(math.log2(calculated_block_size))

    current_block_size = config_data.get('block_size', 768)

    # Always update to the optimal block_size based on actual data
    if recommended_block_size != current_block_size:
        print(f"\nUpdating block_size based on actual data analysis:")
        print(f"  Max nodes in dataset: {max_nodes}")
        print(f"  Max edges in dataset: {max_edges}")
        print(f"  Calculated required tokens: {calculated_block_size}")
        print(f"  Optimal block_size (rounded to power of 2): {recommended_block_size}")
        print(f"  Previous block_size: {current_block_size}")
        print(f"  Setting block_size to {recommended_block_size}...")
        config_data['block_size'] = recommended_block_size
    else:
        print(f"\nBlock size already optimal: {current_block_size}")
        print(f"  Max nodes: {max_nodes}, Max edges: {max_edges}")
        print(f"  Required tokens: {calculated_block_size}")

    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"\nSuccessfully updated configuration file: '{config_path}'")
    except Exception as e:
        print(f"Error writing updates to YAML file {config_path}: {e}")

    return config_data


def create_tokenizer(model_config, dataset_config, tokenizer_path):
    """
    Generates and saves tokenizer files and returns the final vocabulary size.
    Follows the G2PT tokenizer format where:
    - vocab.json contains only non-special tokens (structure + IDX + node/edge types)
    - added_tokens in tokenizer.json contains special tokens with IDs after main vocab
    - vocab_size = total tokens (main vocab + special tokens)
    """
    print("\n--- Starting Tokenizer Creation ---")

    # --- Step 1: Assemble the main vocabulary (without special tokens) ---
    main_vocab = []

    # Add base structural tokens from the model config
    main_vocab.extend(model_config.get('tokens', {}).get('structure', []))

    # Add dynamic index tokens based on max_node_count from the dataset config
    max_nodes = dataset_config.get('max_node_count', 0)
    main_vocab.extend([f"IDX_{i}" for i in range(max_nodes + 1)])

    # Add dataset-specific tokens (node types, edge types)
    main_vocab.extend(dataset_config.get('tokens', {}).get('node_types', []))
    main_vocab.extend(dataset_config.get('tokens', {}).get('edge_types', []))

    # Create the token-to-id mapping for the main vocabulary (non-special tokens only)
    # This will be saved to vocab.json and also used in tokenizer.json's model.vocab
    main_vocab_map = {token: i for i, token in enumerate(main_vocab)}

    # --- Step 2: Determine IDs for special tokens ---
    # Special tokens get IDs starting after the main vocabulary
    special_tokens = model_config.get('tokens', {}).get('special', [])
    special_tokens_map = {}
    next_id = len(main_vocab_map)
    for token in special_tokens:
        special_tokens_map[token] = next_id
        next_id += 1

    # The final vocabulary size includes both main vocab AND special tokens
    vocab_size = len(main_vocab_map) + len(special_tokens_map)
    print(f"Main vocabulary: {len(main_vocab_map)} tokens")
    print(f"Special tokens: {len(special_tokens_map)} tokens")
    print(f"Total vocabulary size: {vocab_size} tokens")

    # --- Step 3: Create tokenizer directory and save files ---
    os.makedirs(tokenizer_path, exist_ok=True)
    print(f"Saving tokenizer files to: {tokenizer_path}")

    # Save vocab.json (main vocabulary only, compact format)
    # This matches the G2PT format: single line, no pretty printing
    with open(os.path.join(tokenizer_path, 'vocab.json'), 'w') as f:
        json.dump(main_vocab_map, f, separators=(',', ': '))

    # Save tokenizer.json (with main vocabulary in model.vocab and special tokens in added_tokens)
    # The model.vocab contains the same content as vocab.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": special_tokens_map.get(token),
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            }
            for token in special_tokens if token in special_tokens_map
        ],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
        "model": {
            "type": "WordLevel",
            "vocab": main_vocab_map,  # Same as vocab.json - no special tokens here
            "unk_token": "[UNK]"
        }
    }
    with open(os.path.join(tokenizer_path, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer_json, f, indent=2)

    # Save tokenizer_config.json (matching G2PT format with BertTokenizer)
    tokenizer_config_json = {
        "added_tokens_decoder": {
            str(v): {
                "content": k,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
            for k, v in special_tokens_map.items()
        },
        "clean_up_tokenization_spaces": False,
        "do_lower_case": False,
        "extra_special_tokens": {},
        "mask_token": "[MASK]",
        "model_max_length": dataset_config.get('block_size', 768),
        "pad_token": "[PAD]",
        "strip_accents": None,
        "tokenizer_class": "BertTokenizer",  # Changed from PreTrainedTokenizerFast
        "unk_token": "[UNK]"
    }
    with open(os.path.join(tokenizer_path, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config_json, f, indent=2)

    print("Tokenizer creation complete.")
    print(f"  - Main vocabulary (vocab.json): {len(main_vocab_map)} tokens")
    print(f"  - Special tokens (added_tokens): {len(special_tokens_map)} tokens")
    print(f"  - Total vocabulary size: {vocab_size}")
    return vocab_size


def main():
    """Main execution function."""
    print(f"--- Starting AIG Dataset Setup ---")

    # Step 1: Analyze AIGER files and update the dataset config
    updated_dataset_config = run_analysis_and_update_config(DATASET_CONFIG_PATH)

    # Step 2: Read the base model config
    try:
        with open(MODEL_CONFIG_PATH, 'r') as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Model config file not found at '{MODEL_CONFIG_PATH}'")
        sys.exit(1)

    # Step 3: Create the tokenizer and get the vocabulary size
    vocab_size = create_tokenizer(model_config, updated_dataset_config, TOKENIZER_OUTPUT_PATH)

    # Step 4: Add the vocabulary size back to the dataset config file
    if vocab_size is not None:
        print(f"\nAdding vocab_size ({vocab_size}) to '{DATASET_CONFIG_PATH}'...")
        updated_dataset_config['vocab_size'] = vocab_size
        try:
            with open(DATASET_CONFIG_PATH, 'w') as f:
                yaml.dump(updated_dataset_config, f, default_flow_style=False, sort_keys=False)
            print("Successfully added vocab_size to config.")
        except Exception as e:
            print(f"Error writing vocab_size to YAML file: {e}")

    print("\n--- AIG Dataset Setup Finished ---")


if __name__ == '__main__':
    # To run this script, ensure you have PyYAML installed:
    # pip install PyYAML
    main()
