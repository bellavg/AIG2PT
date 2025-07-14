import os
import re
import yaml
import sys
import json

# --- Configuration ---
# Path to the dataset-specific config file (will be read and updated)
DATASET_CONFIG_PATH = '../configs/aig.yaml'
# Path to the base model config file (will be read only)
MODEL_CONFIG_PATH = '../configs/network.yaml'  # Corrected to network.yaml
# Hardcoded path for saving the generated tokenizer files
TOKENIZER_OUTPUT_PATH = './tokenizer'


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

    # Use 'raw_data_dir' for analysis, which should be in the config
    raw_aig_directory = config_data.get('raw_data_dir')
    if not raw_aig_directory:
        print(f"Error: 'raw_data_dir' not found in '{config_path}'. This is needed for analysis.")
        sys.exit(1)

    if not os.path.isabs(raw_aig_directory):
        config_file_dir = os.path.dirname(os.path.abspath(config_path))
        raw_aig_directory = os.path.normpath(os.path.join(config_file_dir, '..', raw_aig_directory))

    final_stats = analyze_aig_directory(raw_aig_directory)
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


def create_tokenizer(model_config, dataset_config, tokenizer_path):
    """
    Generates and saves tokenizer files and returns the final vocabulary size.
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

    # Create the token-to-id mapping for the main vocabulary. This will be saved to vocab.json.
    main_vocab_map = {token: i for i, token in enumerate(main_vocab)}

    # --- Step 2: Determine IDs for special tokens ---
    # Special tokens are added *after* the main vocabulary to ensure they have unique, higher IDs.
    special_tokens = model_config.get('tokens', {}).get('special', [])
    special_tokens_map = {}
    next_id = len(main_vocab_map)
    for token in special_tokens:
        special_tokens_map[token] = next_id
        next_id += 1

    # The final vocabulary size includes the special tokens
    vocab_size = len(main_vocab_map) + len(special_tokens_map)
    print(f"Vocabulary size: {vocab_size} tokens")

    # --- Step 3: Create tokenizer directory and save files ---
    os.makedirs(tokenizer_path, exist_ok=True)
    print(f"Saving tokenizer files to: {tokenizer_path}")

    # Save vocab.json (main vocabulary only)
    with open(os.path.join(tokenizer_path, 'vocab.json'), 'w') as f:
        json.dump(main_vocab_map, f, indent=2)

    # Save tokenizer.json (with main vocabulary in model.vocab and special tokens in added_tokens)
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
            "vocab": main_vocab_map,  # Use the main map here
            "unk_token": "[UNK]"
        }
    }
    # Filter out null values from added_tokens before writing
    tokenizer_json['added_tokens'] = [t for t in tokenizer_json['added_tokens'] if t['id'] is not None]
    with open(os.path.join(tokenizer_path, 'tokenizer.json'), 'w') as f:
        json.dump(tokenizer_json, f, indent=2)

    # Save tokenizer_config.json
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
