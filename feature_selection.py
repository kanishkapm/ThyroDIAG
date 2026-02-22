#!/usr/bin/env python3
"""
================================================================================
Stage 3: LASSO Feature Selection
================================================================================
Performs feature selection using LASSO regression to identify
discriminative biomarker genes.

Usage:
    python3 stage3_lasso_feature_selection.py

Author: Kanishka P M
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Configuration
PROJECT_DIR = Path("/home/kanishka/Downloads/thyroid_project2.0")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("="*70)
    print("STAGE 3: LASSO FEATURE SELECTION")
    print("="*70)

    # Load combined data
    data_file = PROCESSED_DIR / "thyroid_combined.csv"

    if not data_file.exists():
        print("✗ No processed data found. Run stage2_preprocess.py first.")
        return False

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file, index_col=0)

    y = df['Subtype']
    X = df.drop(['Subtype', 'Source'], axis=1, errors='ignore')

    # Ensure only numeric columns
    X = X.select_dtypes(include=[np.number])

    print(f"\nDataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {y.value_counts().to_dict()}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    print(f"\nClass encoding: {dict(zip(classes, range(len(classes))))}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Binary classification: Cancer vs Normal for feature selection
    y_binary = (y != 'Normal').astype(int)

    print("\n" + "-"*50)
    print("Stage 3a: LASSO Feature Selection (Binary)")
    print("-"*50)

    # LASSO for feature selection
    lasso = LassoCV(
        alphas=np.logspace(-4, 1, 100),
        cv=5,
        max_iter=10000,
        random_state=42
    )
    lasso.fit(X_scaled, y_binary)

    # Get selected genes
    coefs = pd.Series(lasso.coef_, index=X.columns)
    selected_genes = coefs[coefs != 0].sort_values(key=abs, ascending=False)

    print(f"\nLASSO selected {len(selected_genes)} / {X.shape[1]} genes")

    # Save LASSO results
    lasso_results = pd.DataFrame({
        'Gene': selected_genes.index,
        'Coefficient': selected_genes.values,
        'Abs_Coefficient': np.abs(selected_genes.values)
    })
    lasso_results.to_csv(RESULTS_DIR / "lasso_selected_genes.csv", index=False)
    print(f"✓ Saved: {RESULTS_DIR / 'lasso_selected_genes.csv'}")

    # Get top genes
    top_genes = selected_genes.head(50)
    print(f"\nTop 20 selected genes:")
    for i, (gene, coef) in enumerate(top_genes.head(20).items(), 1):
        direction = "↑" if coef > 0 else "↓"
        print(f"  {i:2}. {gene:<12} {direction} {coef:+.4f}")

    # Multiclass classification with selected genes
    print("\n" + "-"*50)
    print("Stage 3b: Multiclass LASSO Classification")
    print("-"*50)

    # Use selected genes
    X_sel = X[selected_genes.index]
    X_sel_scaled = scaler.fit_transform(X_sel)

    # Logistic Regression with L1
    clf = LogisticRegressionCV(
        Cs=20,
        cv=5,
        penalty='l1',
        solver='saga',
        max_iter=2000,
        random_state=42
    )
    clf.fit(X_sel_scaled, y_encoded)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_sel_scaled, y_encoded, cv=cv, scoring='accuracy')

    print(f"\nMulticlass LASSO CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save results
    results = {
        'n_original_genes': int(X.shape[1]),
        'n_selected_genes': int(len(selected_genes)),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'top_genes': selected_genes.head(50).index.tolist()
    }

    with open(RESULTS_DIR / "lasso_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ LASSO Feature Selection Complete!")
    print(f"  Selected genes: {len(selected_genes)}")
    print(f"  CV Accuracy: {cv_scores.mean():.3f}")

    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
