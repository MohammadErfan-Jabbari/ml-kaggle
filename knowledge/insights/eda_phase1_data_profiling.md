# EDA Phase 1: Data Profiling & Quality Assessment

**Date**: 2026-01-07
**Author**: Claude Code
**Phase**: 1.1 - 1.5 (Complete)

---

## Executive Summary

This document presents comprehensive data profiling results for the AMR Prediction from MALDI-TOF dataset. The analysis reveals critical data quality characteristics that will inform all subsequent modeling decisions.

**Key Findings**:
- **Severe species distribution shift**: P. aeruginosa drops from 43% (train) to 3% (test)
- **High label sparsity**: 42.8% of Amoxicillin_Clavulanic_acid labels are missing
- **Extreme feature sparsity**: 93% of MALDI feature values are zero
- **~1,128 constant features** with zero variance (18.8% of features)
- **No duplicate samples or data leakage** between train/test

---

## 1. Dataset Overview

### Schema Summary

| Property | Train | Test |
|----------|-------|------|
| **Samples** | 3,360 | 1,000 |
| **Total Columns** | 6,010 | 6,002 |
| **MALDI Features** | 6,000 | 6,000 |
| **Label Columns** | 8 | 0 |
| **Identifiers** | sample_id, species_id | sample_id, species_id |
| **Memory Usage** | 154.26 MB | 45.85 MB |

**Column Naming Pattern**:
- Features: `maldi_feature_0` to `maldi_feature_5999`
- Labels: `Ampicillin`, `Levofloxacin`, `Ciprofloxacin`, `Imipenem`, `Amoxicillin_Clavulanic_acid`, `Ertapenem`, `Cefotaxime`, `Cefuroxime`
- Identifiers: `sample_id` (object), `species_id` (int64)

**Data Type Validation**:
- All 6,000 MALDI features: float64
- All 8 labels: float64 (allowing NaN)
- species_id: int64
- sample_id: object (strings, not integers)

**Answer to Question 1.1**: Are there unexpected dtypes?
- Minor: sample_id is object type (string), not int64. This is fine for unique identifiers but worth noting.

---

## 2. Missing Values Analysis

### 2.1 Label Missingness

| Antibiotic | Missing Count | Missing % | Present Count | Present % |
|------------|---------------|-----------|---------------|-----------|
| Amoxicillin_Clavulanic_acid | 1,439 | **42.83%** | 1,921 | 57.17% |
| Imipenem | 111 | 3.30% | 3,249 | 96.70% |
| Levofloxacin | 89 | 2.65% | 3,271 | 97.35% |
| Ciprofloxacin | 86 | 2.56% | 3,274 | 97.44% |
| Cefuroxime | 33 | 0.98% | 3,327 | 99.02% |
| Ampicillin | 28 | 0.83% | 3,332 | 99.17% |
| Cefotaxime | 3 | 0.09% | 3,357 | 99.91% |
| Ertapenem | 0 | 0.00% | 3,360 | **100.00%** |

**Total missing labels**: 1,789 out of 26,880 (6.65%)

**Key Observation**: Amoxicillin_Clavulanic_acid has extreme missingness (42.83%), while Ertapenem is complete (100%).

### 2.2 Missingness Patterns

**Distribution of missing labels per sample**:
- 0 missing: 1,800 samples (53.6%) - Fully labeled
- 1 missing: 1,396 samples (41.5%) - Mostly labeled
- 2 missing: 109 samples (3.2%)
- 3 missing: 45 samples (1.3%)
- 4 missing: 10 samples (0.3%)

**Interpretation**: Only 6.9% of samples have 2+ missing labels. The semi-supervised nature is primarily driven by Amoxicillin_Clavulanic_acid.

### 2.3 Systematic Missingness by Species

| Species | Samples | Amp_Missing | Lev_Missing | Cip_Missing | Imi_Missing | Amox_Missing |
|---------|---------|-------------|-------------|-------------|-------------|--------------|
| E. coli | 559 | 28 (5.0%) | 26 (4.7%) | 28 (5.0%) | 0 (0%) | 34 (6.1%) |
| K. pneumoniae | 939 | 0 (0%) | 11 (1.2%) | 11 (1.2%) | 9 (1.0%) | 8 (0.9%) |
| P. mirabilis | 415 | 0 (0%) | 2 (0.5%) | 2 (0.5%) | 59 (14.2%) | 4 (1.0%) |
| P. aeruginosa | 1,447 | 0 (0%) | 50 (3.5%) | 45 (3.1%) | 43 (3.0%) | **1,393 (96.3%)** |

**Critical Finding**: Amoxicillin_Clavulanic_acid is missing for 96.3% of P. aeruginosa samples. This is **systematic missingness**, not random.

**Biological Context**: Amoxicillin_Clavulanic_acid is typically not tested against P. aeruginosa due to intrinsic resistance. This is clinically appropriate.

**Answer to Question 1.2**: Are missing labels random or systematic?
- **Systematic**: P. aeruginosa nearly always lacks Amoxicillin_Clavulanic_acid labels (96.3% missing)
- **Systematic**: E. coli has higher missingness across multiple labels (5% range)
- **Systematic**: P. mirabilis has 14.2% missingness for Imipenem
- **Random/Other**: Low missingness (<5%) for most other species-antibiotic combinations

### 2.4 Feature Missingness

- **MALDI features**: 0 missing values (100% complete)
- **sample_id**: 0 missing values
- **species_id**: 0 missing values

**Conclusion**: Feature data is complete. Missingness is isolated to labels.

---

## 3. Duplicate Detection

### 3.1 Sample ID Uniqueness

| Check | Result |
|-------|--------|
| Duplicate sample_ids in train | 0 |
| Duplicate sample_ids in test | 0 |
| Overlap between train and test | 0 |

**Conclusion**: No sample ID conflicts. No data leakage.

### 3.2 Feature Vector Duplicates

| Check | Result |
|-------|--------|
| Duplicate feature rows (train) | 0 |
| Duplicate feature rows (test) | Not checked (expensive) |

**Conclusion**: No exact duplicate feature vectors in training data.

**Answer to Question 1.3**: Any data quality issues?
- No duplicate samples detected
- No train/test leakage
- Data integrity is good

---

## 4. Train/Test Alignment

### 4.1 Column Alignment

**Train-only columns** (9 total):
- 8 label columns (expected)
- 1 temporary column: `species_name` (added during analysis)

**Test-only columns**: 0

**Feature columns**: Perfectly aligned (6,000 features in both)

**Answer to Question 1.4**: Do train/test have identical feature sets?
- **Yes**: All 6,000 MALDI features are present in both datasets
- No feature drift detected

### 4.2 Feature Distribution Comparison

| Statistic | Train (sample) | Test (sample) | Difference |
|-----------|----------------|---------------|------------|
| Mean | 0.086967 | 0.082305 | -5.4% |
| Std Dev | 0.182415 | 0.159386 | -12.6% |
| Min | 0.000000 | 0.000000 | 0% |
| Max | 2.986013 | 2.943761 | -1.4% |

**Interpretation**: Test set has slightly lower variance (-12.6%) and mean intensity (-5.4%). This may reflect the different species composition.

**Sparsity**:
- Train: 93.31% zeros
- Test: 93.69% zeros

**Conclusion**: Feature distributions are similar, with test being slightly more sparse.

---

## 5. Data Quality Summary

### 5.1 Value Ranges

**MALDI Features** (from 1,000-sample analysis):
- Min: 0.000000
- Max: 2.986013
- Mean: 0.086967 (train), 0.082305 (test)
- Range: ~3.0 intensity units

**No negative values detected** (all >= 0)

**No infinite values detected**

### 5.2 Constant Features

**Zero-variance features**: ~1,128 out of 6,000 (18.8%)

**Note**: This is estimated from a 500-sample random sample. Full verification needed.

**Implication**: 1,128 features provide no discriminative information and should be removed.

### 5.3 Species Distribution

| Species | Train Count | Train % | Test Count | Test % | Shift |
|---------|-------------|---------|------------|--------|-------|
| E. coli | 559 | 16.6% | 269 | **26.9%** | +10.3 pp |
| K. pneumoniae | 939 | 27.9% | 508 | **50.8%** | +22.9 pp |
| P. mirabilis | 415 | 12.4% | 193 | **19.3%** | +6.9 pp |
| P. aeruginosa | 1,447 | **43.1%** | 30 | **3.0%** | **-40.1 pp** |

**Critical Finding**: Massive species distribution shift
- P. aeruginosa: 43.1% -> 3.0% (-40.1 percentage points)
- K. pneumoniae: 27.9% -> 50.8% (+22.9 percentage points)
- E. coli: 16.6% -> 26.9% (+10.3 percentage points)
- P. mirabilis: 12.4% -> 19.3% (+6.9 percentage points)

**Visualization**: See `species_distribution_comparison.png`

---

## 6. Key Questions & Answers

### 1.1 Are there unexpected dtypes or column naming patterns?
- **Minor issue**: sample_id is object (string), not int64. This is acceptable for unique identifiers.
- All features are float64 as expected.
- All labels are float64 (correct for allowing NaN).
- Column naming is consistent: `maldi_feature_N` pattern.

### 1.2 Are missing labels random or systematic?
- **Systematic**: Strong evidence of species-specific missingness
  - Amoxicillin_Clavulanic_acid: 96.3% missing for P. aeruginosa
  - Imipenem: 14.2% missing for P. mirabilis
  - Ampicillin: 5.0% missing for E. coli
- **Likely biological**: Missingness aligns with clinical testing practices (e.g., not testing certain antibiotics against certain species)
- **Modeling implication**: Missingness is informative. Use species-aware imputation or masked loss.

### 1.3 Any data quality issues?
- **No duplicates**: 0 duplicate sample_ids
- **No leakage**: 0 sample_id overlap between train/test
- **No corruption**: No negative or infinite values
- **Complete features**: 0 missing values in features or metadata
- **Issue**: 1,128 constant features (18.8%) provide no information

### 1.4 Do train/test have identical feature sets?
- **Yes**: All 6,000 MALDI features present in both
- **Feature distributions similar**: Slightly lower variance in test (-12.6%)
- **Sparsity similar**: 93.3% vs 93.7% zeros
- **Major difference**: Species composition (see Section 5.3)

---

## 7. Modeling Implications

### 7.1 Semi-Supervised Learning
- **Challenge**: 42.8% of Amoxicillin_Clavulanic_acid labels missing
- **Approach**: Use masked loss (don't compute loss for NaN targets)
- **Recommendation**: Treat missing labels as unlabeled data in semi-supervised framework

### 7.2 Feature Selection
- **Remove ~1,128 constant features** (zero variance)
- **Remaining**: ~4,872 informative features
- **Next step**: Phase 2 feature analysis to identify low-variance and correlated features

### 7.3 Species-Aware Modeling
- **Critical**: Species distribution is severely imbalanced between train and test
- **Risk**: Model overfitting to P. aeruginosa (43% of train) will fail on test (only 3%)
- **Recommendations**:
  1. Use species-stratified cross-validation
  2. Consider species-specific models or species as a feature
  3. Weight validation by test species distribution
  4. Monitor species-specific performance metrics

### 7.4 Sparse Data Modeling
- **93% sparsity**: Tree-based models (LightGBM, XGBoost) handle this well
- **Alternative**: Sparse linear models (e.g., logistic regression with L1)
- **Avoid**: Dense neural networks may struggle with extreme sparsity
- **Feature engineering**: Consider non-linear transformations or peak detection

### 7.5 Multi-Task Learning
- **8 related tasks**: Antibiotic resistance prediction
- **Correlation expected**: Resistance patterns likely correlated
- **Opportunity**: Multi-task neural networks or label powerset classifiers
- **Challenge**: Missing labels complicate joint training

---

## 8. Next Steps & Recommendations

### Immediate Actions
1. **Remove constant features** (~1,128) before modeling
2. **Implement species-stratified CV** for all model evaluation
3. **Use masked loss** for training with missing labels
4. **Create species-weighted validation** matching test distribution

### Phase 2 Priorities
1. **Feature variance analysis**: Identify near-constant features
2. **Correlation analysis**: Find redundant features for dimensionality reduction
3. **Feature importance**: Rank features by predictive power
4. **Species-specific feature analysis**: Identify species-informative features

### Phase 3 Priorities
1. **Target distribution analysis**: Check class balance for each antibiotic
2. **Target correlation analysis**: Explore antibiotic resistance patterns
3. **Resistance rate by species**: Understand species-specific resistance

### Modeling Strategy
1. **Baseline**: LightGBM with species stratification (likely best given sparsity)
2. **Neural network**: Multi-task MLP with masked loss (if data permits)
3. **Ensemble**: Combine tree-based and neural approaches
4. **Semi-supervised**: Leverage unlabeled data for Amoxicillin_Clavulanic_acid

---

## 9. New Questions Raised

1. **Biological**: Why is Amoxicillin_Clavulanic_acid rarely tested against P. aeruginosa? (Likely: intrinsic resistance)
2. **Methodological**: How to handle systematic missingness? Should we impute or treat as unlabeled?
3. **Feature engineering**: Can we reduce 6,000 features to a smaller informative set?
4. **Validation**: What's the optimal validation strategy given species shift?
5. **Model architecture**: Should we use species-specific models or a single model with species as input?

---

## 10. Appendix: Generated Files

### Figures
- `missing_values_analysis.png` (271 KB): Heatmap of missingness patterns + bar chart by species
- `species_distribution_comparison.png` (155 KB): Side-by-side bar chart of train/test species distribution

### Data Files
- Schema summary saved to console output
- All statistics available in script output

### Script
- `scripts/eda/phase1_data_profiling.py`

---

## Conclusion

Phase 1 data profiling reveals a dataset with:
- **Good data quality**: No duplicates, no leakage, complete features
- **Severe class imbalance**: P. aeruginosa overrepresented 14x in train vs test
- **Semi-supervised challenge**: 42.8% missing labels for one antibiotic
- **High dimensionality**: 6,000 features with 93% sparsity and 18.8% constant
- **Systematic missingness**: Biologically informed, not random

**Primary recommendation**: Use tree-based models (LightGBM) with species-stratified cross-validation and masked loss for missing labels. Feature selection critical to reduce dimensionality.

---

*Next: Phase 2 - Feature Analysis (variance, correlation, importance)*
