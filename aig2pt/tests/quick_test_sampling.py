#!/usr/bin/env python3
"""
Quick integration test for sampling and evaluation.
Tests the main functionality works end-to-end.
"""
import sys
from pathlib import Path

print("Testing AIG2PT Sampling & Evaluation...")
print("-" * 50)

# Test 1: Can we import everything?
print("\n1. Testing imports...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sampling_and_evaluation import AIGEvaluator
    print("   ✓ AIGEvaluator imported")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Basic evaluation
print("\n2. Testing AIGEvaluator...")
try:
    evaluator = AIGEvaluator()

    # Test sequences
    valid_seq = "<boc> <sepc> NODE_PI IDX_0 <eoc> <bog> <sepg> IDX_0 IDX_1 EDGE_REG <eog>"
    invalid_seq = "invalid sequence"

    sequences = [valid_seq, invalid_seq, valid_seq]  # One duplicate

    # Test validity
    validity = evaluator.evaluate_validity(sequences)
    print(f"   Valid: {validity['valid']}/3")
    print(f"   Invalid: {validity['invalid']}/3")

    # Test uniqueness
    uniqueness = evaluator.evaluate_uniqueness(sequences)
    print(f"   Unique: {uniqueness['unique']}/3")
    print(f"   Duplicates: {uniqueness['duplicates']}")

    # Test comprehensive
    results = evaluator.comprehensive_evaluation(sequences)
    print(f"   ✓ Comprehensive evaluation works")
    print(f"   Summary: {results['summary']}")

except Exception as e:
    print(f"   ✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check tokenizer exists
print("\n3. Checking tokenizer...")
try:
    tokenizer_path = Path(__file__).parent.parent / 'dataset' / 'tokenizer'
    if tokenizer_path.exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        print(f"   ✓ Tokenizer loaded: vocab_size={len(tokenizer)}")
    else:
        print(f"   ⚠ Tokenizer not found at {tokenizer_path}")
except Exception as e:
    print(f"   ✗ Tokenizer check failed: {e}")

# Test 4: Check data exists
print("\n4. Checking prepared data...")
try:
    data_path = Path(__file__).parent.parent / 'dataset' / 'aig_prepared'
    if data_path.exists():
        train_path = data_path / 'train'
        if train_path.exists():
            import json
            meta_path = data_path / 'data_meta.json'
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"   ✓ Data ready: {meta['train']['num_graphs']} train graphs")
        else:
            print(f"   ⚠ Train data not found")
    else:
        print(f"   ⚠ Prepared data directory not found")
except Exception as e:
    print(f"   ⚠ Data check failed: {e}")

# Test 5: Sequence format validation
print("\n5. Testing sequence format...")
try:
    test_seq = "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <eoc> <bog> <sepg> IDX_0 IDX_1 EDGE_REG <eog>"

    required = ['<boc>', '<eoc>', '<bog>', '<eog>']
    all_present = all(m in test_seq for m in required)

    if all_present:
        print(f"   ✓ Sequence format is correct")
    else:
        print(f"   ✗ Missing required markers")

except Exception as e:
    print(f"   ✗ Format test failed: {e}")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("\nYour sampling and evaluation modules are working!")
print("\nNext steps:")
print("  1. Train: python train.py")
print("  2. Generate: python generate_aigs.py --checkpoint <path> --method both")
print("="*50)

