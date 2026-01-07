# Phase 3 Target Analysis - Summary

## What Was Done

Implemented comprehensive target variable analysis for the AMR prediction competition, analyzing all 8 antibiotic resistance labels.

## Script Location
`/sata_disk/users/erfan/ml_kaggle/scripts/eda/phase3_target_analysis.py`

Run with:
```bash
uv run python scripts/eda/phase3_target_analysis.py
```

## Key Findings

### 1. Class Balance
- **Ampicillin**: 88.2% resistant (most imbalanced)
- **Amoxicillin_Clavulanic_acid**: 29.9% resistant (least imbalanced)
- Range: 29.9% - 88.2% resistance rates

### 2. Missing Labels
- **Amoxicillin_Clavulanic_acid**: 42.8% missing (major issue)
- **Ertapenem**: 0% missing (complete)
- 7 of 8 antibiotics have some missing labels
- Total: 1,789 missing label instances

### 3. Correlations
- **Fluoroquinolones** (Levofloxacin/Ciprofloxacin): r=0.925 (expected)
- **Carbapenems** (Imipenem/Ertapenem): r=0.772 (expected)
- **Cephalosporins** (Cefotaxime/Cefuroxime): r=0.657 (expected)
- Strong cross-class: Ertapenem/Cefotaxime: r=0.813

### 4. Co-Resistance Patterns
- P(Ciprofloxacin|Levofloxacin) = 97.4% (nearly perfect)
- P(Ertapenem|Imipenem) = 80.8%
- P(Imipenem|Ertapenem) = 96.8%

### 5. Multi-Drug Resistance
- Average: 4.4 resistances per sample
- **MDR (>=3 drugs)**: 68.6% of samples
- **XDR (>=6 drugs)**: 42.0% of samples
- Most common pattern: 1111X111 (resistant to 6/7 tested)

## Deliverables

### Figures (6 PNG files in `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase3/`)
1. `class_balance.png` - Bar chart of resistance rates
2. `missing_labels_pattern.png` - Missing label analysis
3. `target_correlation.png` - 8x8 correlation heatmap
4. `conditional_resistance.png` - P(B|A) heatmap
5. `antibiotic_dendrogram.png` - Hierarchical clustering
6. `resistance_count_distribution.png` - MDR histogram

### Report
`/sata_disk/users/erfan/ml_kaggle/knowledge/insights/eda_phase3_target_analysis.md`
- 9.2 KB comprehensive markdown report
- Includes modeling recommendations
- Code snippets for masked loss
- Architecture suggestions

## Modeling Recommendations

1. **Use Masked Loss** - Critical for handling NaN labels
2. **Multi-Task Architecture** - Shared encoder with 8 task heads
3. **Class Weighting** - For highly imbalanced targets (Ampicillin)
4. **Task Grouping** - Consider shared heads for correlated antibiotics
5. **Validation** - Per-antibiotic AUC with missing label masking

## Biological Insights

- Correlations match known drug class relationships
- Fluoroquinolones show near-perfect correlation (97% co-resistance)
- Carbapenems show strong but asymmetric correlation
- High prevalence of MDR (68.6%) suggests resistant strains are common

## Next Steps

1. Implement `MaskedBCEWithLogitsLoss` in training code
2. Build baseline multi-task MLP
3. Set up proper cross-validation with species stratification
4. Train baseline and establish per-antibiotic AUC benchmarks
5. Compare NN vs LightGBM approaches

