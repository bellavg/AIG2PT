#!/usr/bin/env python3
"""Simple test - run from aig2pt/ directory"""
import sys
from pathlib import Path

# Setup paths
current = Path.cwd()
package_root = Path(__file__).resolve().parents[1]
repo_root = package_root.parent
sys.path.insert(0, str(repo_root))

print(f"Running from: {current}")
print(f"Python: {sys.executable}")
print("="*60)

# Test 1: Evaluator
print("\nTest 1: Evaluator")
print("-"*40)
try:
    from aig2pt.sampling_and_evaluation import AIGEvaluator
    e = AIGEvaluator()
    seqs = [
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
        "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_0 IDX_2 EDGE_REG "
        "<sepg> IDX_1 IDX_2 EDGE_INV <sepg> IDX_2 IDX_3 EDGE_REG <eog>",
        "bad",
        "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
        "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_1 IDX_2 EDGE_REG "
        "<sepg> IDX_0 IDX_2 EDGE_INV <sepg> IDX_2 IDX_3 EDGE_INV <eog>",
    ]
    v = e.evaluate_validity(seqs)
    u = e.evaluate_uniqueness(seqs, validity_mask=v['validity_mask'])
    print(f"Valid: {v['valid']}/3 ✓" if v['valid'] == 2 else f"Valid: {v['valid']}/3 ✗")
    print(
        f"Unique: {u['unique']}/{u['total']} ✓"
        if (u['unique'], u['total']) == (2, 2)
        else f"Unique: {u['unique']}/{u['total']} ✗"
    )
    test1 = v['valid'] == 2 and u['unique'] == 2
except Exception as e:
    print(f"ERROR: {e}")
    test1 = False

# Test 2: Tokenizer
print("\nTest 2: Tokenizer")
print("-"*40)
try:
    from transformers import AutoTokenizer
    tok_path = package_root / 'dataset' / 'tokenizer'
    print(f"Looking at: {tok_path}")
    if tok_path.exists():
        tok = AutoTokenizer.from_pretrained(str(tok_path))
        actual_size = len(tok)
        expected_size = 72
        print(f"Vocab size: {actual_size}")
        test2 = actual_size == expected_size
        print("✓" if test2 else f"✗ Expected {expected_size}, got {actual_size}")
    else:
        print(f"✗ Path does not exist")
        test2 = False
except Exception as e:
    print(f"ERROR: {e}")
    test2 = False

# Test 3: Data
print("\nTest 3: Prepared Data")
print("-"*40)
try:
    import json
    data_path = package_root / 'dataset' / 'aig_prepared' / 'data_meta.json'
    print(f"Looking at: {data_path}")
    if data_path.exists():
        with open(data_path) as f:
            meta = json.load(f)
        n = meta['train']['num_graphs']
        print(f"Train graphs: {n}")
        test3 = n > 0
        print("✓" if test3 else "✗")
    else:
        print(f"✗ Path does not exist")
        test3 = False
except Exception as e:
    print(f"ERROR: {e}")
    test3 = False

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'✓ PASS' if test1 else '✗ FAIL'}  Evaluator")
print(f"{'✓ PASS' if test2 else '✗ FAIL'}  Tokenizer")
print(f"{'✓ PASS' if test3 else '✗ FAIL'}  Data")
print("="*60)

if all([test1, test2, test3]):
    print("\n🎉 ALL TESTS PASSED!\n")
    print("You're ready to:")
    print("  1. Train: python train.py")
    print("  2. Generate: python generate_aigs.py --checkpoint <path> --method both")
else:
    print("\n⚠ Some tests failed")

print("="*60)

