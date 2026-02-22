# Thyroid Cancer Classification - Project 2.0

## Final Results Report

### Project Overview
Automated pipeline for thyroid cancer classification using GEO datasets with ensemble machine learning.

---

## Dataset Summary

| Parameter | Value |
|-----------|-------|
| Total Samples | 479 |
| Original Features | 331 genes |
| Selected Features | 82 genes (LASSO) |
| GEO Datasets | 6 |
| Cancer Subtypes | 4 (Normal, PTC, FTC, ATC) |

---

## Model Performance

### Cross-Validation Results
| Model | Test Accuracy | CV Mean | CV Std |
|-------|-------------|---------|--------|
| LASSO | 100.0% | 100.0% | 0.0% |
| ElasticNet | 100.0% | 100.0% | 0.0% |
| Random Forest | 100.0% | 100.0% | 0.0% |
| SVM | 100.0% | 100.0% | 0.0% |
| KNN | 100.0% | 100.0% | 0.0% |
| **Ensemble ★** | **100.0%** | **100.0%** | **0.0%** |
| Gradient Boosting | 98.6% | 100.0% | 0.0% |

---

## Top Biomarker Genes (LASSO Selected)

### Thyroid-Specific Markers
1. **DIO1** - Deiodinase 1 (thyroid hormone activation)
2. **NIS** (SLC5A5) - Sodium-iodide symporter
3. **TSHR** - TSH receptor
4. **TG** - Thyroglobulin
5. **TPO** - Thyroid peroxidase
6. **DIO2** - Deiodinase 2
7. **SLC26A4** - Pendrin
8. **HNF1A** - Hepatocyte nuclear factor
9. **THRB** - Thyroid hormone receptor beta

### Cancer-Related Genes
- **BRAF** - MAPK pathway (PTC driver mutation)
- **PPARG** - Peroxisome proliferator-activated receptor
- **STAT3** - Signal transducer
- **FN1** - Fibronectin (EMT marker)
- **ZEB2** - EMT transcription factor
- **PIK3CB** - PI3K pathway

---

## GSEA Pathway Enrichment

### Top Enriched Pathways

| Pathway | Database | P-value | -log10(p) |
|---------|----------|---------|------------|
| Thyroid hormone biosynthesis | Reactome | 0.0001 | 4.0 |
| Thyroid cancer | KEGG | 0.0010 | 3.0 |
| hormone metabolic process | GO | 0.0020 | 2.7 |
| PI3K-Akt signaling pathway | KEGG | 0.0050 | 2.3 |
| Signal Transduction | Reactome | 0.0030 | 2.5 |
| thyroid follicle development | GO | 0.0040 | 2.4 |
| MAPK signaling pathway | KEGG | 0.0100 | 2.0 |
| Epithelial mesenchymal transition | Hallmark | 0.0200 | 1.7 |

---

## Key Findings

1. **High Classification Accuracy**: All models achieve near-perfect classification of thyroid cancer subtypes
2. **Key Biomarkers Identified**: 82 genes selected, including established thyroid markers (DIO1, NIS, TG, TPO)
3. **Pathway Enrichment**: Significant enrichment in:
   - Thyroid hormone biosynthesis
   - PI3K-Akt signaling (cancer progression)
   - MAPK signaling (BRAF pathway)
   - Epithelial-mesenchymal transition (metastasis)

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/thyroid_combined.csv` | Combined GEO dataset |
| `results/lasso_selected_genes.csv` | 82 selected biomarker genes |
| `results/ensemble_results.json` | Ensemble model metrics |
| `results/model_comparison.csv` | Model comparison results |
| `results/gsea_enrichment.csv` | Pathway enrichment results |
| `models/ensemble_model.pkl` | Trained ensemble model |

---

## Visualization Files

| File | Description |
|------|-------------|
| `results/model_comparison.png` | Bar chart comparing all models |
| `results/gsea_barplot.png` | Pathway enrichment visualization |

---

## Technical Details

- **Python Version**: 3.8+
- **Key Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Machine Learning**: LASSO (feature selection), Ensemble stacking (RF + SVM + LR)
- **Validation**: 5-fold cross-validation, 80/20 train-test split

---

## Author

**Kanishka P M**
MSc Bioinformatics

Date: February 2026
