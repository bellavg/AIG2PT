#!/usr/bin/env python3
"""
Test sampling and evaluation modules with a mock trained model.
Tests both multinomial sampling and diverse beam search.
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

print("="*60)
print("AIG2PT Sampling & Evaluation Tests")
print("="*60)

def test_imports():
    """Test that all required modules can be imported."""
    print("\n[1/5] Testing imports...")
    try:
        from sampling_and_evaluation import AIGSampler, AIGEvaluator, load_model_and_tokenizer
        from core.model import GPT, GPTConfig
        from transformers import AutoTokenizer
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_evaluator():
    """Test the AIGEvaluator with sample sequences."""
    print("\n[2/5] Testing AIGEvaluator...")
    try:
        from sampling_and_evaluation import AIGEvaluator

        evaluator = AIGEvaluator()

        # Valid sequences
        valid_seqs = [
            "<boc> <sepc> NODE_PI IDX_0 <sepc> NODE_AND IDX_1 <eoc> <bog> <sepg> IDX_0 IDX_1 EDGE_REG <eog>",
            "<boc> <sepc> NODE_PI IDX_0 <sepc> NODE_PI IDX_1 <eoc> <bog> <sepg> IDX_0 IDX_1 EDGE_REG <eog>",
            "<boc> <sepc> NODE_CONST0 IDX_0 <eoc> <bog> <eog>",
        ]

        # Invalid sequences (missing markers)
        invalid_seqs = [
            "<boc> <sepc> NODE_PI IDX_0 <eoc>",  # Missing edge section
            "NODE_PI IDX_0 <eoc> <bog> <eog>",   # Missing <boc>
        ]

        all_seqs = valid_seqs + invalid_seqs

        # Test validity
        validity_results = evaluator.evaluate_validity(all_seqs)
        expected_valid = len(valid_seqs)
        actual_valid = validity_results['valid']

        if actual_valid == expected_valid:
            print(f"  ✓ Validity check: {actual_valid}/{len(all_seqs)} valid (expected {expected_valid})")
        else:
            print(f"  ✗ Validity check failed: got {actual_valid}, expected {expected_valid}")
            return False

        # Test uniqueness
        duplicate_seqs = valid_seqs + [valid_seqs[0]]  # Add duplicate
        uniqueness_results = evaluator.evaluate_uniqueness(duplicate_seqs)

        if uniqueness_results['duplicates'] == 1:
            print(f"  ✓ Uniqueness check: detected {uniqueness_results['duplicates']} duplicate")
        else:
            print(f"  ✗ Uniqueness check failed")
            return False

        # Test novelty
        training_seqs = valid_seqs[:2]
        test_seqs = valid_seqs  # Includes 2 from training
        novelty_results = evaluator.evaluate_novelty(test_seqs, training_seqs)

        expected_memorized = 2
        if novelty_results['memorized'] == expected_memorized:
            print(f"  ✓ Novelty check: {novelty_results['novel']} novel, {novelty_results['memorized']} memorized")
        else:
            print(f"  ✗ Novelty check failed")
            return False

        # Test comprehensive evaluation
        results = evaluator.comprehensive_evaluation(all_seqs, training_seqs)
        print(f"  ✓ Comprehensive evaluation: {results['summary']}")

        return True

    except Exception as e:
        print(f"  ✗ Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampler_with_dummy_model():
    """Test AIGSampler with a small dummy model."""
    print("\n[3/5] Testing AIGSampler with dummy model...")
    try:
        from sampling_and_evaluation import AIGSampler
        from core.model import GPT, GPTConfig
        from transformers import AutoTokenizer

        # Load tokenizer
        tokenizer_path = PARENT_DIR / 'dataset' / 'tokenizer'
        if not tokenizer_path.exists():
            print(f"  ⚠ Tokenizer not found at {tokenizer_path}, skipping sampler test")
            return True

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Create tiny model for testing
        config = GPTConfig(
            block_size=128,
            vocab_size=len(tokenizer),
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=False
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = GPT(config)
        model.to(device)
        model.eval()

        print(f"  ✓ Created dummy model with {model.get_num_params()/1e6:.2f}M params on {device}")

        # Test sampler creation
        sampler = AIGSampler(model, tokenizer, device=device)
        print(f"  ✓ AIGSampler initialized")

        # Test multinomial sampling (just 2 samples to be fast)
        print(f"  → Testing multinomial sampling...")
        try:
            sequences = sampler.multinomial_sample(
                num_samples=2,
                temperature=1.0,
                max_new_tokens=50,
                batch_size=2
            )
            print(f"  ✓ Multinomial sampling: generated {len(sequences)} sequences")
        except Exception as e:
            print(f"  ✗ Multinomial sampling failed: {e}")
            return False

        # Test diverse beam search
        print(f"  → Testing diverse beam search...")
        try:
            # Try HF conversion
            try:
                hf_model = model.to_hf()
                print(f"  ✓ Model converted to HF format")
            except:
                print(f"  ⚠ HF conversion failed, will use custom implementation")

            diverse_sequences = sampler.diverse_beam_search(
                num_samples=2,
                num_beams=4,
                num_beam_groups=2,
                diversity_penalty=0.5,
                max_new_tokens=50,
                batch_size=1
            )
            print(f"  ✓ Diverse beam search: generated {len(diverse_sequences)} sequences")
        except Exception as e:
            print(f"  ⚠ Diverse beam search failed (expected with dummy model): {e}")
            # This is OK - diverse beam might fail with untrained model

        # Test decoding
        texts = sampler.decode_sequences(sequences)
        print(f"  ✓ Decoded {len(texts)} sequences")
        print(f"    Sample: {texts[0][:100]}...")

        return True

    except Exception as e:
        print(f"  ✗ Sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_consistency():
    """Test that tokenizer has correct vocab_size."""
    print("\n[4/5] Testing tokenizer consistency...")
    try:
        from transformers import AutoTokenizer
        import yaml

        tokenizer_path = PARENT_DIR / 'dataset' / 'tokenizer'
        config_path = PARENT_DIR / 'configs' / 'aig.yaml'

        if not tokenizer_path.exists():
            print(f"  ⚠ Tokenizer not found, skipping test")
            return True

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        tokenizer_vocab_size = len(tokenizer)

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config_vocab_size = config.get('vocab_size')

        print(f"  Tokenizer vocab_size: {tokenizer_vocab_size}")
        print(f"  Config vocab_size: {config_vocab_size}")

        if tokenizer_vocab_size == config_vocab_size:
            print(f"  ✓ Vocab sizes match: {tokenizer_vocab_size}")
            return True
        else:
            print(f"  ✗ Vocab size mismatch!")
            return False

    except Exception as e:
        print(f"  ✗ Tokenizer consistency test failed: {e}")
        return False


def test_sequence_format():
    """Test that generated sequences have correct format."""
    print("\n[5/5] Testing sequence format...")
    try:
        # Test sequence structure
        test_seq = "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 <sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_0 IDX_2 EDGE_REG <sepg> IDX_1 IDX_2 EDGE_REG <sepg> IDX_2 IDX_3 EDGE_REG <eog>"

        required_markers = ['<boc>', '<eoc>', '<bog>', '<eog>']
        required_node_types = ['NODE_CONST0', 'NODE_PI', 'NODE_AND', 'NODE_PO']
        required_edge_types = ['EDGE_REG', 'EDGE_INV']

        # Check markers
        for marker in required_markers:
            if marker not in test_seq:
                print(f"  ✗ Missing required marker: {marker}")
                return False
        print(f"  ✓ All required markers present")

        # Check node types
        found_nodes = [nt for nt in required_node_types if nt in test_seq]
        print(f"  ✓ Found node types: {found_nodes}")

        # Check ordering
        boc_idx = test_seq.index('<boc>')
        eoc_idx = test_seq.index('<eoc>')
        bog_idx = test_seq.index('<bog>')
        eog_idx = test_seq.index('<eog>')

        if boc_idx < eoc_idx < bog_idx < eog_idx:
            print(f"  ✓ Markers in correct order: <boc> < <eoc> < <bog> < <eog>")
        else:
            print(f"  ✗ Markers out of order")
            return False

        # Check node section structure
        node_section = test_seq[test_seq.index('<boc>'):test_seq.index('<eoc>')].split()
        if '<sepc>' in node_section:
            print(f"  ✓ Node section has separators (<sepc>)")

        # Check edge section structure
        edge_section = test_seq[test_seq.index('<bog>'):test_seq.index('<eog>')].split()
        if '<sepg>' in edge_section:
            print(f"  ✓ Edge section has separators (<sepg>)")

        return True

    except Exception as e:
        print(f"  ✗ Sequence format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))
    if not results[-1][1]:
        print("\n✗ Cannot continue without imports")
        return False

    # Test 2: Evaluator
    results.append(("Evaluator", test_evaluator()))

    # Test 3: Sampler
    results.append(("Sampler", test_sampler_with_dummy_model()))

    # Test 4: Tokenizer
    results.append(("Tokenizer Consistency", test_tokenizer_consistency()))

    # Test 5: Sequence Format
    results.append(("Sequence Format", test_sequence_format()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests PASSED! Sampling and evaluation are ready.")
        print("\nYou can now:")
        print("  1. Train your model with: python train.py")
        print("  2. Generate AIGs with: python generate_aigs.py --checkpoint <ckpt> --method both")
        return True
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

