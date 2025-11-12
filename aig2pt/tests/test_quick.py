#!/usr/bin/env python3
"""Run this to test sampling and evaluation modules."""

if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Get useful paths
    tests_dir = Path(__file__).resolve().parent
    package_root = tests_dir.parent
    repo_root = package_root.parent

    # Add to path for imports
    sys.path.insert(0, str(repo_root))

    results = []

    # Test 1: Evaluator
    print("\n[1/3] Testing Evaluator...")
    try:
        from aig2pt.sampling_and_evaluation import AIGEvaluator
        evaluator = AIGEvaluator()

        valid_seq_1 = (
            "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
            "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_0 IDX_2 EDGE_REG "
            "<sepg> IDX_1 IDX_2 EDGE_INV <sepg> IDX_2 IDX_3 EDGE_REG <eog>"
        )
        valid_seq_2 = (
            "<boc> <sepc> NODE_CONST0 IDX_0 <sepc> NODE_PI IDX_1 <sepc> NODE_AND IDX_2 "
            "<sepc> NODE_PO IDX_3 <eoc> <bog> <sepg> IDX_1 IDX_2 EDGE_REG "
            "<sepg> IDX_0 IDX_2 EDGE_INV <sepg> IDX_2 IDX_3 EDGE_INV <eog>"
        )

        test_seqs = [valid_seq_1, "missing markers", valid_seq_2]

        validity = evaluator.evaluate_validity(test_seqs)
        uniqueness = evaluator.evaluate_uniqueness(test_seqs, validity_mask=validity['validity_mask'])

        print(f"  Valid: {validity['valid']}/3 (expected 2)")
        print(f"  Unique: {uniqueness['unique']}/{uniqueness['total']} (expected 2/2)")

        passed = validity['valid'] == 2 and uniqueness['unique'] == 2
        results.append(("Evaluator", passed))
        if passed:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
    except Exception as e:
        results.append(("Evaluator", False))
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Tokenizer
    print("\n[2/3] Testing Tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer_path = package_root / 'dataset' / 'tokenizer'
        print(f"  Path: {tokenizer_path}")
        print(f"  Exists: {tokenizer_path.exists()}")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        vocab_size = len(tokenizer)
        print(f"  Vocab size: {vocab_size} (expected 72)")

        passed = vocab_size == 72
        results.append(("Tokenizer", passed))
        if passed:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
    except Exception as e:
        results.append(("Tokenizer", False))
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Data
    print("\n[3/3] Testing Data...")
    try:
        import json
        meta_path = package_root / 'dataset' / 'aig_prepared' / 'data_meta.json'
        print(f"  Path: {meta_path}")
        print(f"  Exists: {meta_path.exists()}")

        with open(meta_path) as f:
            meta = json.load(f)
        num_graphs = meta['train']['num_graphs']
        print(f"  Train graphs: {num_graphs}")

        passed = num_graphs > 0
        results.append(("Data", passed))
        if passed:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
    except Exception as e:
        results.append(("Data", False))
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Print results
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")

    all_passed = all(r[1] for r in results)
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour sampling & evaluation setup is working!")
        print("\nNext steps:")
        print("  1. Train: python train.py")
        print("  2. Generate: python generate_aigs.py --checkpoint <path> --method both")
    else:
        print("⚠ SOME TESTS FAILED - see errors above")
    print("="*60)
    sys.exit(0 if all_passed else 1)

