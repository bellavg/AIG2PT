#!/usr/bin/env python3
"""
Smoke test for AIG2PT training pipeline.
Tests that the full training loop works end-to-end with local data.
"""
import sys
from pathlib import Path
import torch
import json
import numpy as np

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

print("="*60)
print("AIG2PT Training Smoke Test")
print("="*60)

# Import TokenizedAIGDataset class
from torch.utils.data import Dataset

class TokenizedAIGDataset(Dataset):
    """Loads pre-tokenized AIG sequences from binary memmap files."""
    def __init__(self, data_dir, pad_token_id=0):
        self.data_dir = Path(data_dir)

        # Load metadata
        meta_path = self.data_dir.parent / 'data_meta.json'
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        split_name = self.data_dir.name
        self.shape = tuple(metadata[split_name]['token_ids_shape'])
        self.num_graphs = metadata[split_name]['num_graphs']
        self.pad_token_id = pad_token_id

        # Load token IDs as memmap
        self.token_ids = np.memmap(
            self.data_dir / 'token_ids.bin',
            dtype=np.int16,
            mode='r',
            shape=self.shape
        )

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        token_ids = torch.from_numpy(self.token_ids[idx].astype(np.int64))
        attention_mask = (token_ids != self.pad_token_id).long()
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': token_ids.clone()
        }

def test_imports():
    """Test that all required modules can be imported."""
    print("\n[1/6] Testing imports...")
    try:
        from core.model import GPT, GPTConfig
        from transformers import AutoTokenizer
        import yaml
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_tokenizer():
    """Test that the tokenizer can be loaded."""
    print("\n[2/6] Testing tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer_path = SCRIPT_DIR.parent / 'dataset' / 'tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print(f"  ✓ Tokenizer loaded: vocab_size={len(tokenizer)}")
        return True, tokenizer
    except Exception as e:
        print(f"  ✗ Tokenizer load failed: {e}")
        return False, None

def test_dataset():
    """Test that the dataset can be loaded."""
    print("\n[3/6] Testing dataset...")
    try:
        data_dir = PARENT_DIR / 'dataset' / 'aig_prepared' / 'train'

        if not data_dir.exists():
            print(f"  ✗ Data directory not found: {data_dir}")
            return False, None

        dataset = TokenizedAIGDataset(str(data_dir), pad_token_id=0)
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")

        # Test getting a sample
        sample = dataset[0]
        print(f"  ✓ Sample shape: {sample['input_ids'].shape}")
        return True, dataset
    except Exception as e:
        print(f"  ✗ Dataset load failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_init():
    """Test that the model can be initialized."""
    print("\n[4/6] Testing model initialization...")
    try:
        from core.model import GPT, GPTConfig
        from transformers import AutoTokenizer

        # Get actual vocab size from tokenizer
        tokenizer_path = PARENT_DIR / 'dataset' / 'tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        vocab_size = len(tokenizer)

        config = GPTConfig(
            block_size=1024,
            vocab_size=vocab_size,  # Use actual vocab size from tokenizer (currently expected 72)
            n_layer=2,  # Small for testing
            n_head=2,
            n_embd=128,
            dropout=0.0,
            bias=False
        )

        model = GPT(config)
        num_params = model.get_num_params() / 1e6
        print(f"  ✓ Model initialized: {num_params:.2f}M parameters (vocab_size={vocab_size})")
        return True, model
    except Exception as e:
        print(f"  ✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_forward_pass(model, dataset):
    """Test that a forward pass works."""
    print("\n[5/6] Testing forward pass...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # Get a batch
        sample = dataset[0]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)  # Add batch dim
        labels = sample['labels'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

        # Forward pass
        X = input_ids[:, :-1]
        Y = labels[:, 1:]
        Y_mask = attention_mask[:, 1:]

        logits, loss = model(X, Y, Y_mask)

        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {X.shape}")
        print(f"    Output shape: {logits.shape}")
        print(f"    Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step(model, dataset):
    """Test that a training step works."""
    print("\n[6/6] Testing training step...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.train()

        # Setup optimizer
        optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-4,
            betas=(0.9, 0.95),
            device_type='cuda' if device == 'cuda' else 'cpu'
        )

        # Get a batch
        sample = dataset[0]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        labels = sample['labels'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)

        X = input_ids[:, :-1]
        Y = labels[:, 1:]
        Y_mask = attention_mask[:, 1:]

        # Training step
        optimizer.zero_grad()
        logits, loss = model(X, Y, Y_mask)
        loss.backward()
        optimizer.step()

        print(f"  ✓ Training step successful")
        print(f"    Loss: {loss.item():.4f}")

        # Do one more step to verify gradients work
        optimizer.zero_grad()
        logits, loss2 = model(X, Y, Y_mask)
        loss2.backward()
        optimizer.step()

        print(f"    Loss after step 2: {loss2.item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all smoke tests."""
    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))
    if not results[-1][1]:
        print("\n✗ Smoke test FAILED: Cannot import required modules")
        return False

    # Test 2: Tokenizer
    success, tokenizer = test_tokenizer()
    results.append(("Tokenizer", success))
    if not success:
        print("\n✗ Smoke test FAILED: Cannot load tokenizer")
        return False

    # Test 3: Dataset
    success, dataset = test_dataset()
    results.append(("Dataset", success))
    if not success:
        print("\n✗ Smoke test FAILED: Cannot load dataset")
        return False

    # Test 4: Model
    success, model = test_model_init()
    results.append(("Model Init", success))
    if not success:
        print("\n✗ Smoke test FAILED: Cannot initialize model")
        return False

    # Test 5: Forward pass
    success = test_forward_pass(model, dataset)
    results.append(("Forward Pass", success))
    if not success:
        print("\n✗ Smoke test FAILED: Forward pass failed")
        return False

    # Test 6: Training step
    success = test_training_step(model, dataset)
    results.append(("Training Step", success))
    if not success:
        print("\n✗ Smoke test FAILED: Training step failed")
        return False

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All smoke tests PASSED! Training pipeline is ready.")
        return True
    else:
        print("\n✗ Some tests FAILED. Please fix the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

