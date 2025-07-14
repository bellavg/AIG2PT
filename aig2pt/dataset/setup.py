import os
import re
import yaml
import sys
import json

# --- Configuration ---
# Path to the dataset-specific config file (will be read and updated)
DATASET_CONFIG_PATH = 'configs/aig.yaml'
# Path to the base model config file (will be read only)
MODEL_CONFIG_PATH = 'configs/model.yaml'


def parse_aiger_header(file_path):
    """
    Reads the first line of an AIGER file and parses the header.
    """
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline()
            matches = re.findall(r'\d+', header_line)

            if header_line.startswith('aig') and len(matches) >= 5:
                return {
                    'max_node_count': int(matches[0]),
                    'pi_count': int(matches[1]),
                    'po_count': int(matches[3]),
                    'and_count': int(matches[4]),
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
    if not os.path.isdir(directory_path):
        print(f"Error: Data directory not found at '{directory_path}'")
        sys.exit(1)

    stats = {
        'max_node_count': 0,
        'pi_counts': {'min': float('inf'), 'max': 0},
        'po_counts': {'min': float('inf'), 'max': 0},
        'and_counts': {'min': float('inf'), 'max': 0},
    }

    file_count = 0
    print(f"Searching for .aag and .aig files in '{directory_path}'...")
    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(('.aag', '.aig')):
                data = parse_aiger_header(os.path.join(dirpath, filename))
                if data:
                    file_count += 1
                    stats['max_node_count'] = max(stats['max_node_count'], data['max_node_count'])
                    stats['pi_counts']['min'] = min(stats['pi_counts']['min'], data['pi_count'])
                    stats['pi_counts']['max'] = max(stats['pi_counts']['max'], data['pi_count'])
                    stats['po_counts']['min'] = min(stats['po_counts']['min'], data['po_count'])
                    stats['po_counts']['max'] = max(stats['po_counts']['max'], data['po_count'])
                    stats['and_counts']['min'] = min(stats['and_counts']['min'], data['and_count'])
                    stats['and_counts']['max'] = max(stats['and_counts']['max'], data['and_count'])

    if file_count == 0:
        print(f"Warning: No AIGER files found in '{directory_path}'.")
        return None

    print(f"Analysis complete. Found and processed {file_count} AIGER files.")
    for key, value in stats.items():
        if isinstance(value, dict) and value['min'] == float('inf'):
            value['min'] = 0
    return stats


def run_analysis_and_update_config(config_path):
    """
    Reads config, analyzes AIGER files, and updates the config with stats.
    Returns the updated config data.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        sys.exit(1)

    aig_directory = config_data.get('raw_data_dir')
    if not aig_directory:
        print(f"Error: 'raw_data_dir' not found in '{config_path}'.")
        sys.exit(1)

    if not os.path.isabs(aig_directory):
        config_file_dir = os.path.dirname(os.path.abspath(config_path))
        aig_directory = os.path.normpath(os.path.join(config_file_dir, aig_directory))

    final_stats = analyze_aig_directory(aig_directory)
    if not final_stats:
        return config_data  # Return original data if no stats were generated

    config_data.update({
        'max_node_count': final_stats['max_node_count'],
        'pi_counts': final_stats['pi_counts'],
        'po_counts': final_stats['po_counts'],
        'and_counts': final_stats['and_counts']
    })

    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"\nSuccessfully updated configuration file: '{config_path}'")
    except Exception as e:
        print(f"Error writing updates to YAML file {config_path}: {e}")

    return config_data


def create_tokenizer(model_config, dataset_config):
    """
    Generates and saves tokenizer files based on combined model and dataset configs.
    """
    print("\n--- Starting Tokenizer Creation ---")

    # 1. Assemble the full vocabulary
    vocab = []

    # Add base structural tokens
    vocab.extend(model_config['tokens']['structure'])

    # Add dynamic index tokens based on max_node_count
    max_nodes = dataset_config['max_node_count']
    vocab.extend([f"IDX_{i}" for i in range(max_nodes + 1)])

    # Add dataset-specific tokens (node types, edge types)
    vocab.extend(dataset_config['tokens']['node_types'])
    vocab.extend(dataset_config['tokens']['edge_types'])

    # Add dynamic PI and PO count tokens
    pi_min, pi_max = dataset_config['pi_counts']['min'], dataset_config['pi_counts']['max']
    po_min, po_max = dataset_config['po_counts']['min'], dataset_config['po_counts']['max']
    vocab.extend([f"PI_COUNT_{i}" for i in range(pi_min, pi_max + 1)])
    vocab.extend([f"PO_COUNT_{i}" for i in range(po_min, po_max + 1)])

    # Create the token-to-id mapping
    vocab_map = {token: i for i, token in enumerate(vocab)}

    # Add special tokens at the end
    special_tokens = model_config['tokens']['special']
    special_tokens_map = {}
    for token in special_tokens:
        token_id = len(vocab_map)
        vocab_map[token] = token_id
        special_tokens_map[token] = token_id

    print(f"Vocabulary size: {len(vocab_map)} tokens")

    # 2. Get tokenizer path and create directory
    tokenizer_path = dataset_config.get('tokenizer_path')
    if not tokenizer_path:
        print("Error: 'tokenizer_path' not defined in dataset config. Cannot save tokenizer.")
        return

    if not os.path.isabs(tokenizer_path):
        config_file_dir = os.path.dirname(os.path.abspath(DATASET_CONFIG_PATH))
        tokenizer_path = os.path.normpath(os.path.join(config_file_dir, tokenizer_path))

    os.makedirs(tokenizer_path, exist_ok=True)
    print(f"Saving tokenizer files to: {tokenizer_path}")

    # 3. Create and save tokenizer files

    # --- vocab.json ---
    with open(os.path.join(tokenizer_path, 'vocab.json'), 'w') as f:
        json.dump(vocab_map, f, indent=2)

    # --- tokenizer.json ---
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": special_tokens_map.get("[UNK]"), "content": "[UNK]", "single_word": False,
                "lstrip": False, "rstrip": False, "normalized": False, "special": True
            },
            {
                "id": special_tokens_map.get("[PAD]"), "content": "[PAD]", "single_word": False,
                "lstrip": False, "rstrip": False, "normalized": False, "special": True
            },
            {
                "id": special_tokens_map.get("[MASK]"), "content": "[MASK]", "single_word": False,
                "lstrip": False, "rstrip": False, "normalized": False, "special": True
            }
        ],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
        "model": {
            "type": "WordLevel",
            "vocab": vocab_map,
            "unk_token": "[UNK]"
        }
    }
    with open(os.path.join(tokenizer_path, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer_json, f, indent=2)

    # --- tokenizer_config.json ---
    tokenizer_config_json = {
        "added_tokens_decoder": {
            str(v): {"content": k, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                     "special": True}
            for k, v in special_tokens_map.items()
        },
        "clean_up_tokenization_spaces": False,
        "do_lower_case": False,
        "model_max_length": model_config.get('block_size', 768),
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "unk_token": "[UNK]",
        "tokenizer_class": "BertTokenizer"
    }
    with open(os.path.join(tokenizer_path, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config_json, f, indent=2)

    print("Tokenizer creation complete.")


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

    # Step 3: Create the tokenizer using both configs
    create_tokenizer(model_config, updated_dataset_config)

    print("\n--- AIG Dataset Setup Finished ---")


if __name__ == '__main__':
    # To run this script, ensure you have PyYAML installed:
    # pip install PyYAML
    main()
