# EDA Phase 6: Train-Test Distribution Analysis

**Date**: 2026-01-07
**Analysis**: Comprehensive distribution shift detection between train and test sets

---

## Executive Summary

This analysis reveals **significant distribution shift** between train and test sets, primarily driven by:
- **Species distribution imbalance**: P. aeruginosa 43% → 3% (train → test)
- **Covariate shift**: Domain classifier AUC = 0.6677
- **Feature-level shifts**: 185/1000 features significantly different

---

## 1. Species Distribution Shift

### Chi-Square Test
- **Statistic**: 555.88
- **P-value**: 3.70e-120
- **Significance**: YES - Significant shift detected

### Species Breakdown
| Species | Train % | Test % | Shift (pp) |
|---------|---------|--------|------------|
| E. coli | 16.6% | 26.9% | +10.3 |
| K. pneumoniae | 27.9% | 50.8% | +22.9 |
| P. mirabilis | 12.4% | 19.3% | +6.9 |
| P. aeruginosa | 43.1% | 3.0% | -40.1 |

**Key Finding**: Largest shift in **P. aeruginosa** species

---

## 2. Feature Distribution Analysis

### Overall Statistics
- **Mean difference** (absolute): 0.051286
- **Mean difference** (percentage): 36082.10%
- **Std difference** (absolute): 0.050417
- **Std difference** (percentage): 1083418.07%

### Kolmogorov-Smirnov Test Results
- **Features tested**: 1000
- **Significant shifts** (p < 0.01): 185
- **Mean KS statistic**: 0.0455

### Top 10 Most Shifted Features
- **maldi_feature_1350**: KS = 0.3954, p = 2.01e-108
- **maldi_feature_1557**: KS = 0.3935, p = 2.64e-107
- **maldi_feature_1559**: KS = 0.3923, p = 1.15e-106
- **maldi_feature_2594**: KS = 0.3820, p = 5.55e-101
- **maldi_feature_538**: KS = 0.3795, p = 1.26e-99
- **maldi_feature_1351**: KS = 0.3756, p = 1.52e-97
- **maldi_feature_1973**: KS = 0.3755, p = 1.66e-97
- **maldi_feature_811**: KS = 0.3755, p = 1.71e-97
- **maldi_feature_445**: KS = 0.3754, p = 1.79e-97
- **maldi_feature_1554**: KS = 0.3746, p = 5.24e-97


---

## 3. Species-Stratified Distribution Shift

| Species | Mean KS Statistic | Max KS Statistic | Train Samples | Test Samples |
|---------|-------------------|------------------|---------------|--------------|
| E. coli | 0.0113 | 0.1084 | 559 | 269 |
| K. pneumoniae | 0.0068 | 0.0745 | 939 | 508 |
| P. mirabilis | 0.0107 | 0.1332 | 415 | 193 |
| P. aeruginosa | 0.0296 | 0.3679 | 1447 | 30 |


**Key Finding**: P. aeruginosa shows highest within-species distribution shift

---

## 4. Covariate Shift Detection

### Domain Classifier Results
- **Algorithm**: Random Forest (100 trees, max_depth=10)
- **Cross-validated AUC**: 0.6677
- **Interpretation**: Significant covariate shift

### Most Discriminative Features
- **maldi_feature_1556**: Importance = 0.0168
- **maldi_feature_1559**: Importance = 0.0115
- **maldi_feature_1553**: Importance = 0.0107
- **maldi_feature_201**: Importance = 0.0101
- **maldi_feature_1555**: Importance = 0.0101
- **maldi_feature_1867**: Importance = 0.0101
- **maldi_feature_2662**: Importance = 0.0095
- **maldi_feature_537**: Importance = 0.0088
- **maldi_feature_1865**: Importance = 0.0086
- **maldi_feature_965**: Importance = 0.0085


---

## 5. Density Estimation & OOD Detection

### Methodology
- **Dimensionality reduction**: PCA (50 components, 67.8% variance)
- **Density estimation**: Kernel Density Estimation (Gaussian kernel)
- **OOD threshold**: 5th percentile of train density

### Results
- **Train log-density** (mean ± std): -4.948 ± 0.928
- **Test log-density** (mean ± std): -7.332 ± 39.403
- **OOD samples**: 7.6% of test set below train threshold

### Interpretation
- Moderate OOD presence - some test samples differ from train


---

## 6. Multivariate Distribution Analysis

### Jensen-Shannon Divergence: 0.2747

**Interpretation**: Significantly different multivariate distributions

### Energy Distance Statistics
- **Mean**: 0.0101
- **Median**: 0.0000
- **Max**: 0.2854

---

## Key Findings Summary

### Critical Issues
1. **Severe species distribution shift** - P. aeruginosa overrepresented in train (43% vs 3%)
2. **Significant covariate shift** - Domain classifier AUC of 0.6677 indicates clear distribution differences
3. **185 features** show statistically significant distribution differences

### Moderate Concerns
1. **7.6% OOD samples** - Some test samples not well-represented in training
2. **Feature mean shifts** - Average 36082.1% difference in feature means

---

## Recommendations

### Immediate Actions
1. **Use species-stratified cross-validation** - Ensure each fold reflects test distribution
2. **Implement domain adaptation** - Consider importance weighting or adversarial training
3. **Monitor per-species performance** - Track metrics separately for each species

### Modeling Strategy
1. **Feature selection** - Remove or downweight features with extreme distribution shift
2. **Ensemble methods** - Use models robust to distribution shift (e.g., tree-based)
3. **Uncertainty estimation** - Identify and flag OOD samples during inference
4. **Species-specific handling** - Consider separate models or weighting for P. aeruginosa

### Validation Strategy
1. **Stratified Group K-Fold** - Group by species to prevent leakage
2. **Domain-aware validation** - Create validation splits matching test distribution
3. **Per-species metrics** - Report AUC for each species separately

### Advanced Techniques
1. **Importance weighting** - Upweight underrepresented species (E. coli, K. pneumoniae)
2. **Domain adversarial training** - Learn features invariant to train-test shift
3. **Test-time adaptation** - Adjust model predictions based on test distribution
4. **Pseudo-labeling** - Leverage test data for semi-supervised learning

---

## Figures Generated

1. `feature_distribution_shift.png` - Histogram overlays of top 12 shifted features
2. `feature_statistics_difference.png` - Distribution of mean and std differences
3. `species_stratified_shift.png` - Per-species feature shift analysis
4. `domain_classifier_feature_importance.png` - Most discriminative features
5. `density_estimation_pca.png` - KDE density visualization and PCA scatter plot

---

## Statistical Tests Summary

| Test | Statistic | P-value | Significant | Interpretation |
|------|-----------|---------|-------------|----------------|
| Chi-square (species) | 555.88 | 3.70e-120 | Yes | Species distribution differs |
| KS test (features) | 0.0455 (mean) | - | Yes | 185 features shifted |
| Domain classifier | AUC = 0.6677 | - | Yes | Covariate shift detected |

---

## Conclusion

The train-test distribution shift in this competition is **significant and multifaceted**:
- **Primary driver**: Species distribution imbalance
- **Secondary factor**: Feature-level distribution differences
- **Impact**: Models may overfit to train distribution, especially P. aeruginosa

**Success depends on**: Proper validation strategy, domain-aware modeling, and careful handling of species imbalance.

---

*Analysis performed using Phase 6 EDA script*
*All figures saved to `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase6/`*
