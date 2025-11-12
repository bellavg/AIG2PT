#!/usr/bin/env python3
"""Run this to test sampling and evaluation modules."""

if __name__ == '__main__':
    import sys
    from pathlib import Path

    # Get the aig2pt directory (where this file is)
    aig2pt_dir = Path(__file__).parent.resolve()

    # Add to path for imports
    sys.path.insert(0, str(aig2pt_dir))

    results = []

    # Test 1: Evaluator
    print("\n[1/3] Testing Evaluator...")
    try:
        from sampling_and_evaluation import AIGEvaluator
        evaluator = AIGEvaluator()

        test_seqs = [
            "<boc> <eoc> <bog> <eog>",
            "missing markers",
            "<boc> <eoc> <bog> <eog>"  # duplicate
        ]

        validity = evaluator.evaluate_validity(test_seqs)
        uniqueness = evaluator.evaluate_uniqueness(test_seqs)

        print(f"  Valid: {validity['valid']}/3 (expected 2)")
        print(f"  Unique: {uniqueness['unique']}/3 (expected 2)")

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
        # Correct path: aig2pt/dataset/tokenizer
        tokenizer_path = aig2pt_dir / 'dataset' / 'tokenizer'
        print(f"  Path: {tokenizer_path}")
        print(f"  Exists: {tokenizer_path.exists()}")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        vocab_size = len(tokenizer)
        print(f"  Vocab size: {vocab_size} (expected 72)")

        passed = vocab_size == 72  # 69 main + 3 special
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
        # Correct path: aig2pt/dataset/aig_prepared/data_meta.json
        meta_path = aig2pt_dir / 'dataset' / 'aig_prepared' / 'data_meta.json'
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

