#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import shutil

print("=" * 60)
print("DEBUG: Starting setup script test")
print("=" * 60)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
print(f"Python path: {sys.path[0]}")

try:
    print("\n1. Importing modules...")
    from aig2pt.dataset import setup
    print("   ✓ Import successful")

    print("\n2. Checking paths...")
    print(f"   DATASET_CONFIG_PATH: {setup.DATASET_CONFIG_PATH}")
    print(f"   MODEL_CONFIG_PATH: {setup.MODEL_CONFIG_PATH}")
    print(f"   TOKENIZER_OUTPUT_PATH: {setup.TOKENIZER_OUTPUT_PATH}")
    print(f"   Config exists: {setup.DATASET_CONFIG_PATH.exists()}")

    # Clean up old tokenizer to force fresh generation
    if setup.TOKENIZER_OUTPUT_PATH.exists():
        print(f"\n3. Removing old tokenizer directory: {setup.TOKENIZER_OUTPUT_PATH}")
        shutil.rmtree(setup.TOKENIZER_OUTPUT_PATH)
        print("   ✓ Old tokenizer removed")

    print("\n4. Running main()...")
    setup.main()
    print("\n✓ Setup completed successfully!")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

