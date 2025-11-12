"""
File structure validation for LayerDAG baseline.

Verifies that all necessary files are in place.
"""
import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists


def validate_structure():
    """Validate the LayerDAG baseline directory structure."""
    print("="*60)
    print("Validating LayerDAG Baseline Structure")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_exist = True
    
    # Check directories
    print("\nDirectories:")
    dirs = ['dataset', 'model', 'configs']
    for d in dirs:
        path = os.path.join(base_dir, d)
        all_exist &= check_file_exists(path, f"{d}/ directory")
    
    # Check dataset files
    print("\nDataset files:")
    dataset_files = [
        'dataset/__init__.py',
        'dataset/aig_layerdag.py',
        'dataset/general.py',
        'dataset/layer_dag.py'
    ]
    for f in dataset_files:
        path = os.path.join(base_dir, f)
        all_exist &= check_file_exists(path, f)
    
    # Check model files
    print("\nModel files:")
    model_files = [
        'model/__init__.py',
        'model/layer_dag.py',
        'model/diffusion.py'
    ]
    for f in model_files:
        path = os.path.join(base_dir, f)
        all_exist &= check_file_exists(path, f)
    
    # Check configuration
    print("\nConfiguration:")
    config_files = ['configs/aig.yaml']
    for f in config_files:
        path = os.path.join(base_dir, f)
        all_exist &= check_file_exists(path, f)
    
    # Check scripts
    print("\nScripts:")
    scripts = [
        'train_layerdag.py',
        'sample_layerdag.py',
        'preprocess_aigs.py',
        'setup_utils.py'
    ]
    for f in scripts:
        path = os.path.join(base_dir, f)
        all_exist &= check_file_exists(path, f)
    
    # Check documentation
    print("\nDocumentation:")
    docs = ['README.md']
    for f in docs:
        path = os.path.join(base_dir, f)
        all_exist &= check_file_exists(path, f)
    
    print("\n" + "="*60)
    if all_exist:
        print("Structure validation: PASSED ✓")
        print("All required files are present.")
    else:
        print("Structure validation: FAILED ✗")
        print("Some files are missing.")
    print("="*60)
    
    return all_exist


def check_imports():
    """Check if Python files have valid syntax."""
    print("\n" + "="*60)
    print("Checking Python File Syntax")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    python_files = [
        'dataset/__init__.py',
        'dataset/aig_layerdag.py',
        'model/__init__.py',
        'setup_utils.py',
        'train_layerdag.py',
        'sample_layerdag.py',
        'preprocess_aigs.py',
    ]
    
    all_valid = True
    for f in python_files:
        path = os.path.join(base_dir, f)
        try:
            with open(path, 'r') as file:
                compile(file.read(), path, 'exec')
            print(f"✓ {f} - syntax OK")
        except SyntaxError as e:
            print(f"✗ {f} - syntax error: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"✗ {f} - file not found")
            all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print("Syntax validation: PASSED ✓")
    else:
        print("Syntax validation: FAILED ✗")
    print("="*60)
    
    return all_valid


if __name__ == '__main__':
    structure_ok = validate_structure()
    syntax_ok = check_imports()
    
    print("\n" + "="*60)
    print("OVERALL VALIDATION")
    print("="*60)
    print(f"Structure: {'PASSED ✓' if structure_ok else 'FAILED ✗'}")
    print(f"Syntax:    {'PASSED ✓' if syntax_ok else 'FAILED ✗'}")
    print("="*60)
    
    sys.exit(0 if (structure_ok and syntax_ok) else 1)
