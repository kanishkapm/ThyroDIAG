#!/usr/bin/env python3
"""
================================================================================
Main Runner - Project 2.0
================================================================================
Runs all stages in sequence.

Usage:
    python3 run_all_stages.py

Author: Kanishka P M
================================================================================
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path("/home/kanishka/Downloads/thyroid_project2.0")

STAGES = [
    ("Stage 1: Download GEO Data", "stage1_download_data.py"),
    ("Stage 2: Preprocess Data", "stage2_preprocess.py"),
    ("Stage 3: LASSO Feature Selection", "stage3_lasso_feature_selection.py"),
    ("Stage 4: Ensemble Model", "stage4_ensemble_model.py"),
    ("Stage 5: Model Comparison", "stage5_model_comparison.py"),
    ("Stage 6: GSEA Analysis", "stage6_gsea_analysis.py"),
]


def run_stage(name, script):
    """Run a single stage."""
    print("\n" + "="*70)
    print(f"RUNNING: {name}")
    print("="*70)

    result = subprocess.run(
        ["python3", script],
        cwd=PROJECT_DIR,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n✗ FAILED: {name}")
        return False

    print(f"\n✓ COMPLETED: {name}")
    return True


def main():
    print("="*70)
    print("THYROID CANCER CLASSIFICATION - PROJECT 2.0")
    print("="*70)

    # Check Python environment
    result = subprocess.run(["python3", "--version"], capture_output=True, text=True)
    print(f"Python: {result.stdout.strip()}")

    # Check virtual environment
    venv_python = PROJECT_DIR / "venv" / "bin" / "python"
    if not venv_python.exists():
        print("\nCreating virtual environment...")
        subprocess.run(["python3", "-m", "venv", "venv"], cwd=PROJECT_DIR)

        # Install packages
        print("Installing packages...")
        subprocess.run([
            str(venv_python), "-m", "pip", "install",
            "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "joblib", "requests"
        ], cwd=PROJECT_DIR)

    # Run stages
    for name, script in STAGES:
        success = run_stage(name, script)
        if not success:
            print(f"\nPipeline stopped at {name}")
            sys.exit(1)

    print("\n" + "="*70)
    print("ALL STAGES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutput files:")
    print(f"  - {PROJECT_DIR / 'results'}")
    print(f"  - {PROJECT_DIR / 'models'}")
    print(f"  - {PROJECT_DIR / 'data' / 'processed'}")


if __name__ == "__main__":
    main()
