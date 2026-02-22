#!/usr/bin/env python3
"""
================================================================================
Stage 4: Ensemble Stacking Model (Simplified)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

PROJECT_DIR = Path("/home/kanishka/Downloads/thyroid_project2.0")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("="*70)
    print("STAGE 4: ENSEMBLE STACKING MODEL")
    print("="*70)

    # Load data
    data_file = PROCESSED_DIR / "thyroid_combined.csv"
    df = pd.read_csv(data_file, index_col=0)
    y = df['Subtype']
    X = df.drop(['Subtype', 'Source'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = le.classes_

    print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")

    # Simple ensemble: Average predictions from 3 models
    print("\nTraining base models...")

    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train_enc)
    rf_proba = rf.predict_proba(X_test_scaled)
    rf_acc = accuracy_score(y_test_enc, rf.predict(X_test_scaled))
    print(f"  Random Forest: {rf_acc:.3f}")

    # Model 2: SVM
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train_enc)
    svm_proba = svm.predict_proba(X_test_scaled)
    svm_acc = accuracy_score(y_test_enc, svm.predict(X_test_scaled))
    print(f"  SVM: {svm_acc:.3f}")

    # Model 3: Logistic Regression
    lr = LogisticRegressionCV(Cs=10, cv=3, max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train_enc)
    lr_proba = lr.predict_proba(X_test_scaled)
    lr_acc = accuracy_score(y_test_enc, lr.predict(X_test_scaled))
    print(f"  Logistic Regression: {lr_acc:.3f}")

    # Ensemble: Average probabilities
    ensemble_proba = (rf_proba + svm_proba + lr_proba) / 3
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = accuracy_score(y_test_enc, ensemble_pred)

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Random Forest:    {rf_acc:.3f}")
    print(f"SVM:             {svm_acc:.3f}")
    print(f"Logistic Reg:    {lr_acc:.3f}")
    print(f"ENSEMBLE ★:      {ensemble_acc:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, le.inverse_transform(ensemble_pred)))

    # Save model
    model_data = {
        'rf': rf, 'svm': svm, 'lr': lr,
        'scaler': scaler, 'label_encoder': le,
        'classes': classes
    }
    joblib.dump(model_data, MODELS_DIR / "ensemble_model.pkl")
    print(f"\n✓ Saved: {MODELS_DIR / 'ensemble_model.pkl'}")

    # Save results
    results = {
        'test_accuracy': float(ensemble_acc),
        'rf_accuracy': float(rf_acc),
        'svm_accuracy': float(svm_acc),
        'lr_accuracy': float(lr_acc)
    }
    with open(RESULTS_DIR / "ensemble_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Stage 4 Complete!")
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
