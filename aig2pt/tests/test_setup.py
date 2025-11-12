#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure repository root is on sys.path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

print("Starting test...")
from aig2pt.dataset.setup import main

if __name__ == '__main__':
    main()

