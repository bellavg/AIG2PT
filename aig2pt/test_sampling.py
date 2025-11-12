#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("AIG2PT Sampling & Evaluation Test")
print("="*60)

# Test 1
print("\n[1/3] Testing Evaluator...")
try:
    from sampling_and_evaluation import AIGEvaluator
    evaluator = AIGEvaluator()

    test_seqs = ["<boc> <eoc> <bog> <eog>", "invalid", "<boc> <eoc> <bog> <eog>"]
    validity = evaluator.evaluate_validity(test_seqs)
    uniqueness = evaluator.evaluate_uniqueness(test_seqs)

    print(f"  Valid: {validity['valid']}/3")
    print(f"  Unique: {uniqueness['unique']}/3")

    if validity['valid'] == 2 and uniqueness['unique'] == 2:
        print("  ✓ PASS")
        test1 = True
    else:
        print("  ✗ FAIL")
        test1 = False
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    test1 = False

# Test 2
print("\n[2/3] Testing Tokenizer...")
try:
    from transformers import AutoTokenizer
    tok_path = Path(__file__).parent / 'dataset' / 'tokenizer'
    print(f"  Looking for: {tok_path}")

    if not tok_path.exists():
        print(f"  ✗ Path does not exist!")
        test2 = False
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
        vocab_size = len(tokenizer)
        print(f"  Vocab size: {vocab_size}")

        if vocab_size == 72:
            print("  ✓ PASS")
            test2 = True
        else:
            print(f"  ✗ FAIL: Expected 72, got {vocab_size}")
            test2 = False
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    test2 = False

# Test 3
print("\n[3/3] Testing Data...")
try:
    import json
    data_path = Path(__file__).parent / 'dataset' / 'aig_prepared' / 'data_meta.json'
    print(f"  Looking for: {data_path}")

    if not data_path.exists():
        print(f"  ✗ Path does not exist!")
        test3 = False
    else:
        with open(data_path) as f:
            meta = json.load(f)
        num_graphs = meta['train']['num_graphs']
        print(f"  Train graphs: {num_graphs}")

        if num_graphs > 0:
            print("  ✓ PASS")
            test3 = True
        else:
            print(f"  ✗ FAIL: No graphs found")
            test3 = False
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    test3 = False

# Summary
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"{'✓' if test1 else '✗'} Evaluator")
print(f"{'✓' if test2 else '✗'} Tokenizer")
print(f"{'✓' if test3 else '✗'} Data")

all_pass = test1 and test2 and test3
print("\n" + ("🎉 ALL TESTS PASSED!" if all_pass else "⚠ SOME TESTS FAILED"))
print("="*60)

sys.exit(0 if all_pass else 1)

