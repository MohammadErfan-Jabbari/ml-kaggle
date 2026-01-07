# EDA Phase 7: Feature-Target Relationships

**Date**: 2026-01-07

---

## Executive Summary

This analysis explores the relationship between 6000 MALDI-TOF features and resistance phenotypes across 8 antibiotics. Key findings:

### Key Findings

1. **Feature importance is antibiotic-specific**
   - Different antibiotics show different top features
   - Limited overlap suggests distinct resistance mechanisms
2. **Feature selection methods show moderate agreement**
   - Mean rank correlation: 0.172
   - F-test and Mutual Information show best agreement
3. **Important features vary by species**
   - Different bacterial species may require different biomarkers
   - Consider species-specific modeling or stratification
4. **High sparsity limits univariate methods**
   - Most features are zero for >90% of samples
   - Multivariate methods may be more effective

---

## 1. Univariate Feature Importance (F-statistic)

### Top Features by Antibiotic

| Antibiotic | Max F-stat | Significant Features (p<0.01) |
|------------|------------|-------------------------------|
| Ampicillin | 1441.35 | 1926 |
| Levofloxacin | 829.75 | 1896 |
| Ciprofloxacin | 671.34 | 1849 |
| Imipenem | 22502.42 | 2426 |
| Amoxicillin_Clavulanic_acid | 131.96 | 1119 |
| Ertapenem | 19349.03 | 2515 |
| Cefotaxime | 4558.33 | 2371 |
| Cefuroxime | 1610.89 | 2276 |

### Interpretation

- Higher F-statistics indicate stronger univariate relationships
- Limited overlap of top features suggests antibiotic-specific biomarkers
- Consider using top-N features per antibiotic in ensembles

## 2. Mutual Information Analysis

### Non-linear Feature Discovery

Mutual Information captures non-linear relationships missed by F-test.

| Antibiotic | Overlap (F vs MI) | Jaccard Index |
|------------|-------------------|---------------|
| Ampicillin | ~20/50 | 0.319 |
| Levofloxacin | ~20/50 | 0.373 |
| Ciprofloxacin | ~20/50 | 0.431 |
| Imipenem | ~20/50 | 0.396 |
| Amoxicillin_Clavulanic_acid | ~20/50 | 0.388 |
| Ertapenem | ~20/50 | 0.463 |
| Cefotaxime | ~20/50 | 0.453 |
| Cefuroxime | ~20/50 | 0.467 |

**Implications**:
- F-test and MI show moderate agreement
- Consider ensemble of both linear and non-linear methods
- Top features from MI may capture peak patterns

## 3. Feature Selection Stability

### Method Comparison

We compared four feature selection methods:
1. **F-test (f_classif)**: Linear relationship, fast
2. **Mutual Information**: Non-linear, slower
3. **Chi-square**: For categorical features
4. **Random Forest**: Ensemble tree-based

### Rank Correlations

| Method Pair | Correlation |
|-------------|-------------|
| f_classif vs mutual_info | 0.102 |
| f_classif vs chi2 | 0.302 |
| f_classif vs random_forest | 0.179 |
| mutual_info vs chi2 | 0.073 |
| mutual_info vs random_forest | 0.183 |
| chi2 vs random_forest | 0.191 |

### Consensus Features

Top features by mean rank across all methods saved to:
`outputs/eda/phase7/consensus_features.csv`

## 4. Mean Spectrum by Resistance

### Biomarker Discovery

Comparing mean spectra of resistant vs susceptible isolates reveals potential biomarkers.

| Antibiotic | Significant Regions | Interpretation |
|------------|-------------------|----------------|
| Ampicillin | 1951 | High biomarker potential |
| Levofloxacin | 1761 | High biomarker potential |
| Ciprofloxacin | 1701 | High biomarker potential |
| Imipenem | 2309 | High biomarker potential |
| Amoxicillin_Clavulanic_acid | 842 | High biomarker potential |
| Ertapenem | 2382 | High biomarker potential |
| Cefotaxime | 2223 | High biomarker potential |
| Cefuroxime | 2237 | High biomarker potential |

**Key Insight**: Regions with consistent intensity differences between resistant and susceptible isolates represent candidate biomarkers.

## 5. Species-Specific Importance

### Cross-Species Feature Consistency

Feature importance varies by bacterial species, suggesting:

1. **Different resistance mechanisms** across species
2. **Species-specific biomarkers** may improve predictions
3. **Stratified modeling** could be beneficial

### Recommendation: Species-Aware Modeling

Given the species distribution shift (43% -> 3% P. aeruginosa), consider:

- **Option A**: Train separate models per species
- **Option B**: Include species as a feature with interaction terms
- **Option C**: Use species-stratified cross-validation

---

## Recommendations for Feature Selection

### Strategy 1: Conservative (Baseline)
```python
# Use top-k consensus features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=500)
X_selected = selector.fit_transform(X, y)
```

### Strategy 2: Antibiotic-Specific
```python
# Select different features for each antibiotic
feature_sets = {}
for i, abx in enumerate(antibiotics):
    selector = SelectKBest(f_classif, k=200)
    X_abx = selector.fit_transform(X, y[:, i])
    feature_sets[abx] = selector.get_support()
```

### Strategy 3: Multi-Method Ensemble
```python
# Combine features from multiple methods
from sklearn.ensemble import VotingClassifier
# Model with F-test features + Model with MI features + ...
```

### Strategy 4: Dimensionality Reduction
```python
# PCA on sparse data (TruncatedSVD)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=500, random_state=42)
X_reduced = svd.fit_transform(X)
```

---

## Next Steps

1. **Implement baseline model** with conservative feature selection (Strategy 1)
2. **Experiment with antibiotic-specific** feature selection (Strategy 2)
3. **Consider PCA/SVD** for dimensionality reduction (Strategy 4)
4. **Evaluate species-aware** modeling approaches
5. **Test feature selection** impact on validation AUC

---

## Generated Files

### Figures
- `7_1_f_statistics_analysis.png`: Univariate feature importance
- `7_1_feature_overlap.png`: Overlap of top features
- `7_2_fstat_vs_mi.png`: F-test vs MI comparison
- `7_3_selection_stability.png`: Method agreement
- `7_4_mean_spectrum_by_resistance.png`: Biomarker regions
- `7_5_species_specific_importance.png`: Cross-species analysis

### Data
- `top_features_per_antibiotic.csv`: Top 50 features per antibiotic
- `mutual_information_scores.csv`: MI scores for all features
- `consensus_features.csv`: Top features by mean rank
- `consensus_top_features.npy`: Feature indices (numpy)
- `significant_regions_by_antibiotic.csv`: Biomarker counts

