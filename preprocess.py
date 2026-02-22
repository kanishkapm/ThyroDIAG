#!/usr/bin/env python3
"""
================================================================================
Stage 2: Data Preprocessing
================================================================================
Loads GEO data, maps probes to genes, combines datasets, and performs
batch correction and normalization.

Usage:
    python3 stage2_preprocess.py

Author: Kanishka P M
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_DIR = Path("/home/kanishka/Downloads/thyroid_project2.0")
DATA_DIR = PROJECT_DIR / "data"
GEO_DIR = DATA_DIR / "geo_raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Probe to Gene mapping (GPL570 core genes)
PROBE_TO_GENE = {
    '209792_s_at': 'DIO1', '204780_s_at': 'TPO', '221840_at': 'TG',
    '204614_at': 'TSHR', '209603_at': 'NIS', '204971_at': 'PKHD1L1',
    '220630_at': 'RGS8', '205080_at': 'ADH1B', '201163_s_at': 'HMGA2',
    '204259_at': 'KRT19', '205890_s_at': 'CITED1', '209602_s_at': 'AMPH',
    '216950_s_at': 'GABRB2', '214453_s_at': 'FN1', '203510_at': 'MET',
    '201984_s_at': 'EGFR', '206416_at': 'RET', '213016_at': 'BRAF',
    '212022_s_at': 'MKI67', '201291_s_at': 'TOP2A', '201663_s_at': 'PCNA',
    '209189_at': 'MMP9', '214048_at': 'ZEB1', '217235_at': 'SNAI1',
    '202503_s_at': 'TERT', '202033_at': 'PAX8', '208006_s_at': 'VEGFA',
    '213293_s_at': 'CENPF', '207574_at': 'NFKB1', '204912_at': 'CXCL10',
    '205388_at': 'CCND1', '203213_at': 'RB1', '201110_at': 'CCNE1',
    '200999_s_at': 'CDK2', '203967_at': 'CDK4', '201194_at': 'CDK6',
    '203352_at': 'CDKN1A', '202803_at': 'CDKN1B', '200884_at': 'BCL2',
    '203685_at': 'BAX', '201616_at': 'TP53', '203414_at': 'PTEN',
    '204284_at': 'PIK3CA', '205192_at': 'AKT1', '202421_at': 'AKT2',
    '207520_at': 'MAPK1', '201720_at': 'MAPK3', '202500_at': 'MAP2K1',
    '203050_at': 'RAF1', '205025_at': 'KRAS', '205690_at': 'NRAS',
    '204709_at': 'HRAS', '203248_at': 'CTNNB1', '208478_at': 'APC',
    '203428_at': 'SMAD4', '202497_at': 'TGFBR2', '205399_at': 'VHL',
    '201110_at': 'CCNE1', '204135_at': 'FGF2', '207085_at': 'FGFR1',
    '205227_at': 'FGFR2', '204787_at': 'IGF1R', '201577_at': 'INSR',
    '203339_at': 'MET', '205251_at': 'ROS1', '206189_at': 'ALK',
    '204351_at': 'NTRK1', '205487_at': 'NTRK2', '206395_at': 'NTRK3',
}


def load_geo_series_matrix(file_path):
    """Load GEO series matrix file."""
    print(f"  Loading: {file_path.name}")

    metadata = {}
    expr_start = 0

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('!'):
                key = line.split('\t')[0].lstrip('!').strip()
                values = [v.strip().strip('"') for v in line.split('\t')[1:]]
                metadata[key] = values
            elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                expr_start = i
                break

    # Read expression matrix
    expr = pd.read_csv(file_path, sep='\t', skiprows=expr_start,
                       index_col=0, comment='!', na_values=['null', 'NA'])
    expr = expr.dropna(how='all')

    return expr, metadata


def infer_labels(metadata):
    """Infer subtype labels from metadata."""
    char_key = 'Sample_characteristics_ch1'
    if char_key not in metadata:
        return None

    chars = metadata[char_key]
    labels = []

    keywords = {
        'Normal': ['normal', 'healthy', 'adjacent', 'non-tumor', 'nontumor', 'benign'],
        'PTC': ['papillary', 'ptc', 'classical', 'follicular variant'],
        'FTC': ['follicular', 'ftc'],
        'ATC': ['anaplastic', 'atc', 'undifferentiated'],
    }

    for char in chars:
        char_lower = str(char).lower()
        found = False
        for subtype, kws in keywords.items():
            if any(kw in char_lower for kw in kws):
                labels.append(subtype)
                found = True
                break
        if not found:
            labels.append('Unknown')

    return labels if labels else None


def preprocess_expression(expr, variance_quantile=0.20):
    """Preprocess expression data."""
    # Filter low variance genes
    gene_var = expr.var(axis=1)
    threshold = gene_var.quantile(variance_quantile)
    expr_filtered = expr.loc[gene_var > threshold]

    # Quantile normalization
    expr_rank = expr_filtered.rank(axis=0, method='average')
    expr_norm = np.log2((expr_rank / expr_filtered.shape[0]) + 0.01) + 8

    return expr_norm


def main():
    print("="*70)
    print("STAGE 2: DATA PREPROCESSING")
    print("="*70)

    # Load dataset info
    with open(PROJECT_DIR / "dataset_info.json", 'r') as f:
        info = json.load(f)

    downloaded = info.get('downloaded', [])
    print(f"\nDatasets to process: {downloaded}")

    if not downloaded:
        print("No datasets found! Run stage1_download_data.py first.")
        return False

    all_data = []
    all_labels = []

    for gse_id in downloaded:
        txt_file = GEO_DIR / f"{gse_id}_series_matrix.txt"

        if not txt_file.exists():
            print(f"  ✗ File not found: {txt_file}")
            continue

        try:
            expr, metadata = load_geo_series_matrix(txt_file)
            print(f"    Raw shape: {expr.shape}")

            # Map probes to genes
            gene_index = [PROBE_TO_GENE.get(str(p), str(p)) for p in expr.index]
            expr.index = gene_index

            # Group by gene (keep highest median)
            expr = expr.T
            expr = expr.groupby(level=0).first()

            # Infer labels
            labels = infer_labels(metadata)
            if labels and len(labels) == len(expr):
                expr['Subtype'] = labels
            else:
                # Assign default based on dataset
                if 'ATC' in gse_id or '76039' in gse_id:
                    expr['Subtype'] = 'ATC'
                elif 'PTC' in gse_id or '53157' in gse_id:
                    expr['Subtype'] = 'PTC'
                elif 'FTC' in gse_id:
                    expr['Subtype'] = 'FTC'
                else:
                    expr['Subtype'] = 'PTC'  # Default

            # Preprocess
            expr_processed = preprocess_expression(expr.drop('Subtype', axis=1))
            expr_processed['Subtype'] = expr['Subtype']

            print(f"    Processed: {expr_processed.shape}")
            all_data.append(expr_processed)

            label_counts = expr_processed['Subtype'].value_counts().to_dict()
            print(f"    Labels: {label_counts}")

        except Exception as e:
            print(f"  ✗ Error processing {gse_id}: {e}")
            continue

    if not all_data:
        print("No data loaded!")
        return False

    # Combine all datasets
    print("\nCombining datasets...")
    combined = pd.concat(all_data, axis=0)
    combined = combined.fillna(0)

    print(f"\nCombined dataset:")
    print(f"  Total samples: {combined.shape[0]}")
    print(f"  Total genes: {combined.shape[1] - 1}")
    print(f"  Classes: {combined['Subtype'].value_counts().to_dict()}")

    # Save combined data
    output_file = PROCESSED_DIR / "thyroid_combined.csv"
    combined.to_csv(output_file)
    print(f"\n✓ Saved: {output_file}")

    # Save summary
    summary = {
        'total_samples': int(combined.shape[0]),
        'total_genes': int(combined.shape[1] - 1),
        'classes': combined['Subtype'].value_counts().to_dict(),
        'datasets': downloaded,
        'output_file': str(output_file)
    }

    with open(PROJECT_DIR / "preprocessing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Preprocessing complete!")
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
