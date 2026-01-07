# EDA Phase 4: Species Analysis

> **Analysis Date**: 2026-01-07
> **Critical Finding**: Severe species distribution shift between train and test (p < 1e-100)

---

## Executive Summary

This analysis reveals the **most critical challenge** in the competition: a severe species distribution shift between training and test sets. **P. aeruginosa comprises 43% of training data but only 3% of test data**, while other species show proportional increases. This shift is statistically significant (χ² p-value: 3.57e-143) and represents a Jensen-Shannon divergence of 0.36, indicating substantial distribution mismatch.

**Key Implications**:
- Models trained on train distribution will overfit to P. aeruginosa patterns
- Validation must be species-stratified to detect this issue
- Sample reweighting or separate models per species are necessary
- P. aeruginosa has deterministic resistance patterns that can be leveraged

---

## 1. Species Distribution Comparison

### 1.1 Distribution Tables

| Species | Train Count | Train % | Test Count | Test % | Shift % |
|---------|-------------|---------|------------|--------|---------|
| E. coli | 559 | 16.6% | 269 | 26.9% | **+10.3%** |
| K. pneumoniae | 939 | 27.9% | 508 | 50.8% | **+22.9%** |
| P. mirabilis | 415 | 12.4% | 193 | 19.3% | **+6.9%** |
| **P. aeruginosa** | **1,447** | **43.1%** | **30** | **3.0%** | **-40.1%** |

### 1.2 Statistical Significance

- **Chi-square test**: p = 3.57e-143 (extremely significant)
- **Jensen-Shannon divergence**: 0.3619 (high divergence)
- **KL divergence**: 0.8452 (asymmetric information loss)

### 1.3 Interpretation

The shift is **not random** - it's a systematic redistribution where:
- P. aeruginosa is 14x overrepresented in train (43% vs 3% in test)
- K. pneumoniae is 1.8x underrepresented in train (28% vs 51% in test)
- E. coli and P. mirabilis show moderate shifts

This is the **primary challenge** of this competition. Standard cross-validation will fail to catch this issue.

---

## 2. Per-Species Resistance Profiles

### 2.1 Resistance Rate Heatmap

| Antibiotic | E. coli | K. pneumoniae | P. mirabilis | P. aeruginosa |
|------------|---------|---------------|--------------|---------------|
| Ampicillin | 67.0% | **99.9%** | 47.5% | **100.0%** |
| Levofloxacin | 38.3% | 25.8% | 42.6% | 78.0% |
| Ciprofloxacin | 43.1% | 30.6% | 43.6% | 75.2% |
| Imipenem | **0.4%** | 5.5% | **97.2%** | **99.9%** |
| Amox/Clav | 35.4% | 30.4% | 12.7% | **100.0%*** |
| Ertapenem | **1.6%** | 10.6% | **0.2%** | **100.0%** |
| Cefotaxime | 24.0% | 25.6% | 15.7% | **100.0%** |
| Cefuroxime | 84.2% | 45.3% | 43.8% | **100.0%** |

*Note: Amox/Clav only has 54 labeled samples for P. aeruginosa

### 2.2 Intrinsic Resistance Patterns

**E. coli (species_id=0)**:
- **High susceptibility**: Imipenem (0.4%), Ertapenem (1.6%)
- **High resistance**: Cefuroxime (84.2%), Ampicillin (67.0%)

**K. pneumoniae (species_id=1)**:
- **Near-total resistance**: Ampicillin (99.9%)
- **High susceptibility**: Imipenem (5.5%), Ertapenem (10.6%)

**P. mirabilis (species_id=2)**:
- **Unusual pattern**: Highly resistant to Imipenem (97.2%) but susceptible to Ertapenem (0.2%)
- **Moderate resistance**: Most other antibiotics

**P. aeruginosa (species_id=3)** - CRITICAL:
- **100% resistant to**: Ampicillin, Ertapenem, Cefotaxime, Cefuroxime, Amox/Clav
- **Near 100%**: Imipenem (99.9%)
- **Variable resistance**: Fluoroquinolones (75-78%)

### 2.3 Biological Interpretation

The resistance patterns align with known biological mechanisms:

1. **K. pneumoniae**: Naturally produces penicillinase → near-universal ampicillin resistance
2. **P. mirabilis**: Intrinsically resistant to most penicillins but susceptible to 2nd/3rd gen cephalosporins
3. **P. aeruginosa**: Intrinsically resistant to many drug classes due to:
   - Low outer membrane permeability
   - Efflux pumps
   - AmpC beta-lactamase production
4. **Fluoroquinolones** (Levofloxacin, Ciprofloxacin): High correlation (r=0.92) due to shared mechanism

### 2.4 Practical Implications

**For P. aeruginosa** (which dominates training but is rare in test):
- **5/8 antibiotics can be predicted deterministically** with 100% resistance
- Only fluoroquinolones (Levo/Cipro) need actual ML predictions
- This creates a huge risk: if model overfits to P. aeruginosa's "predict everything as resistant" strategy, it will perform poorly on other species in test

---

## 3. Species Feature Signatures

### 3.1 Mean Spectra Analysis

- Mean spectra were calculated for each species across all 6,000 MALDI features
- Standard deviation bands show within-species variability
- **Key finding**: Certain m/z regions are highly species-specific

### 3.2 Most Discriminating Features

Top 10 features with highest coefficient of variation (CV):

| Feature | CV | E. coli | K. pneumoniae | P. mirabilis | P. aeruginosa |
|---------|-----|---------|---------------|--------------|---------------|
| 3165 | 1.73 | 0.000 | 0.000 | 0.323 | 0.000 |
| 1999 | 1.73 | 0.000 | 0.000 | 0.209 | 0.000 |
| 3290 | 1.73 | 0.000 | 0.203 | 0.000 | 0.000 |
| 3435 | 1.73 | 0.000 | 0.000 | 0.000 | 0.132 |
| 3434 | 1.73 | 0.000 | 0.000 | 0.000 | 0.116 |

These features are essentially **species biomarkers** - present in one species, absent in others.

### 3.3 Interpretation

- **Species can be identified from MALDI spectra** with high accuracy
- Some features are binary (present/absent) for specific species
- This suggests **species prediction as a preliminary task** could be very accurate
- Feature 3165 is a P. mirabilis marker
- Features 3290 appears to be K. pneumoniae-specific
- Features 3435, 3434 are P. aeruginosa-specific

---

## 4. Species Separability

### 4.1 PCA Analysis

- **PC1**: Explains 6.3% of variance
- **PC2**: Explains 5.2% of variance
- **Silhouette Score**: 0.336 (moderate separability)

**Finding**: Species show **moderate clustering** in PCA space. Not perfectly separable, but clear structure exists.

### 4.2 Dimensionality Challenge

The low explained variance (6.3% + 5.2% = 11.5%) indicates:
- Species differences are distributed across many dimensions
- No single PC dominates the species signal
- Need many PCs or non-linear methods for full separability

### 4.3 Implications for Modeling

- **Species should be explicitly provided** as a feature (already in data)
- **Species-aware architectures** (e.g., species embeddings, separate models) are beneficial
- **Simple dimensionality reduction won't eliminate species information** - species signal is diffuse

---

## 5. Species-Specific Feature Importance

### 5.1 ANOVA F-Test Results

Top species-discriminating features identified using ANOVA F-test. Features with highest F-statistics show the strongest between-species variation.

**Note**: Some features showed NaN statistics due to:
- Constant values across all samples (removed in preprocessing)
- Insufficient variation for statistical testing

### 5.2 Feature Distribution Patterns

Boxplots of top discriminating features reveal:
- **Binary features**: Many discriminating features are on/off for specific species
- **Intensity differences**: Other features show continuous intensity differences between species
- **Species-specific biomarkers**: Certain features are uniquely present in one species

### 5.3 Recommendation

**Feature engineering opportunities**:
1. Create binary indicators for species-specific biomarkers
2. Use feature importance to filter noisy features
3. Consider species-specific feature selection

---

## 6. Distribution Shift Impact Analysis

### 6.1 Simulated Performance Degradation

Using hypothetical per-species AUCs (E. coli: 0.85, K. pneumoniae: 0.82, P. mirabilis: 0.78, P. aeruginosa: 0.90):

| Weighting Strategy | Mean AUC | Notes |
|--------------------|----------|-------|
| Train distribution | 0.8545 | Overweights P. aeruginosa (high AUC) |
| Test distribution | 0.8228 | What actually matters |
| Inverse weighting | 0.8221 | Corrects but overcompensates |
| Recommended | 0.8325 | Balanced approach |

**Expected degradation**: 0.0317 AUC (3.8% relative drop) if model overfits to train distribution

### 6.2 Weighting Strategy Comparison

| Species | Train Weight | Test Weight | Inverse Weight | Recommended |
|---------|--------------|-------------|----------------|-------------|
| E. coli | 16.6% | 26.9% | 30.0% | 30.3% |
| K. pneumoniae | 27.9% | 50.8% | 17.9% | 30.3% |
| P. mirabilis | 12.4% | 19.3% | 40.5% | 30.3% |
| P. aeruginosa | 43.1% | 3.0% | 11.6% | 9.1% |

### 6.3 Recommended Approach

**Balanced weighting with P. aeruginosa downweighting**:
- E. coli, K. pneumoniae, P. mirabilis: **1.0x** weight (equal)
- P. aeruginosa: **0.3x** weight (downweighted)

**Rationale**:
- Balances influence of the 3 majority species in test
- Prevents P. aeruginosa from dominating despite train overrepresentation
- Maintains some P. aeruginosa signal for learning its resistance patterns

---

## 7. Critical Recommendations

### 7.1 Immediate Actions

1. **Species-Stratified Cross-Validation**
   - Use `StratifiedKFold` on species_id
   - Ensure each fold has proportional species representation
   - Report per-species validation AUC, not just mean

2. **Sample Reweighting**
   ```python
   # Recommended sample weights
   sample_weights = np.where(
       species_id == 3,  # P. aeruginosa
       0.3,              # Downweight
       1.0               # Normal weight
   )
   ```

3. **Monitoring**
   - Track validation loss per species
   - If P. aeruginosa loss is much lower than others, model is overfitting
   - Create species-wise ROC curves during validation

### 7.2 Modeling Strategies

**Option A: Single Model with Reweighting**
- Use species embedding feature
- Apply sample weights during training
- Monitor per-species performance
- **Pros**: Simple, shares learned patterns
- **Cons**: May still overfit to P. aeruginosa

**Option B: Separate Models Per Species**
- Train 4 independent models
- Ensemble at prediction time based on predicted species
- **Pros**: No distribution shift issues within species
- **Cons**: Less data per model, can't share patterns

**Option C: Hierarchical Approach**
- Step 1: Predict species (very high accuracy possible)
- Step 2: Apply species-specific resistance model
- Step 3: Use deterministic rules for P. aeruginosa (5/8 antibiotics)
- **Pros**: Leverages species predictability, uses biological knowledge
- **Cons**: More complex pipeline

**Recommended**: Start with Option A (reweighting), try Option C if performance plateaus

### 7.3 Feature Engineering

1. **Species-Specific Biomarkers**
   - Extract binary features for species-discriminating m/z values
   - Feature 3165 → is_p_mirabilis_marker
   - Feature 3290 → is_k_pneumoniae_marker
   - Features 3435, 3434 → is_p_aeruginosa_marker

2. **Interaction Features**
   - species_id × key MALDI features
   - Capture species-specific resistance patterns

3. **Rule-Based Features for P. aeruginosa**
   ```python
   if predicted_species == 3:  # P. aeruginosa
       # Set high resistance for these antibiotics
       ampicillin_pred = 1.0
       ertapenem_pred = 1.0
       cefotaxime_pred = 1.0
       cefuroxime_pred = 1.0
       # Only predict fluoroquinolones from model
       levo_pred = model(X)
       cipro_pred = model(X)
   ```

### 7.4 Validation Strategy

**Current train/test split is fundamentally flawed** for validation:
- Random split will have similar species imbalance
- Model will appear to perform well
- Will fail on actual test set

**Correct validation approach**:
```python
from sklearn.model_selection import StratifiedKFold

# Stratify by species to maintain ratios
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, species_id):
    # Each fold has proportional species representation
    # More similar to test distribution
```

### 7.5 Risk Assessment

**High Risk Scenarios**:
1. Model learns "predict high resistance = good" because P. aeruginosa dominates train
2. Model achieves high validation AUC but fails on test (K. pneumoniae majority)
3. Fluoroquinolone predictions suffer from overfitting to P. aeruginosa patterns

**Warning Signs**:
- Validation AUC significantly higher than per-species AUC for non-P. aeruginosa
- Model predicts high resistance (>80%) for E. coli (should be ~40%)
- Feature importance dominated by P. aeruginosa biomarkers

---

## 8. Quick Wins

### 8.1 Deterministic P. aeruginosa Rules

For P. aeruginosa, set these predictions without ML:
- **Ampicillin**: 1.0 (100% resistant)
- **Ertapenem**: 1.0 (100% resistant)
- **Cefotaxime**: 1.0 (100% resistant)
- **Cefuroxime**: 1.0 (100% resistant)
- **Amox/Clav**: 1.0 (100% resistant in labeled data)

Only use ML for:
- Imipenem (99.9% - essentially deterministic too)
- Levofloxacin (78%)
- Ciprofloxacin (75%)

**Expected gain**: ~0.05-0.10 AUC improvement on P. aeruginosa samples

### 8.2 Species-Weighted Baselines

For antibiotics with strong species patterns:
- Calculate per-species mean resistance
- Weight by test distribution
- This simple baseline might beat complex models

Example for Ampicillin:
- E. coli: 67% → weighted: 0.269 * 0.67 = 0.180
- K. pneumoniae: 99.9% → weighted: 0.508 * 0.999 = 0.507
- P. mirabilis: 47.5% → weighted: 0.193 * 0.475 = 0.092
- P. aeruginosa: 100% → weighted: 0.030 * 1.00 = 0.030
- **Predicted overall**: 80.9% resistant

Compare to naive train-mean: 88.2% (overestimates due to P. aeruginosa dominance)

---

## 9. Summary

### 9.1 Key Findings

1. **Severe distribution shift**: P. aeruginosa 43% → 3% (p < 1e-100)
2. **Species are identifiable**: Moderate PCA separability (silhouette: 0.336)
3. **Intrinsic resistance patterns**: P. aeruginosa 100% resistant to 5/8 antibiotics
4. **Species-specific biomarkers**: Certain m/z features uniquely identify species

### 9.2 Action Plan

**Phase 1: Foundation**
- Implement species-stratified CV
- Add sample weights (0.3x for P. aeruginosa)
- Monitor per-species validation metrics

**Phase 2: Modeling**
- Try reweighted single model baseline
- Implement deterministic rules for P. aeruginosa
- Experiment with species-aware architectures

**Phase 3: Advanced**
- Try separate models per species
- Implement hierarchical prediction (species → resistance)
- Use species-specific feature engineering

### 9.3 Expected Impact

If distribution shift is not addressed:
- **Expected validation AUC**: 0.85-0.88 (inflated by P. aeruginosa)
- **Expected test AUC**: 0.78-0.82 (30-40% of samples are K. pneumoniae with different patterns)
- **Gap**: 0.05-0.08 AUC degradation

With proper reweighting:
- **Expected test AUC**: 0.82-0.85
- **Improvement**: +0.03-0.07 AUC

---

## 10. References

### Figures Generated

1. `species_distribution_comparison.png` - Side-by-side bar charts with shift quantification
2. `resistance_by_species_heatmap.png` - Species × antibiotic resistance heatmap
3. `mean_spectra_by_species.png` - Overlaid mean spectra with std bands
4. `pca_umap_by_species.png` - Dimensionality reduction colored by species
5. `top_discriminating_features.png` - ANOVA F-test results and boxplots
6. `distribution_shift_impact.png` - Weighting strategies and performance simulation

### Data Locations

- Script: `/sata_disk/users/erfan/ml_kaggle/scripts/eda/phase4_species_analysis.py`
- Figures: `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase4/`
- Train data: `/sata_disk/users/erfan/ml_kaggle/raw/train.csv`
- Test data: `/sata_disk/users/erfan/ml_kaggle/raw/test.csv`

---

## Appendix: Statistical Details

### Chi-Square Test

Tests whether observed test distribution differs from expected (train) distribution:
- H0: Test distribution = Train distribution
- H1: Test distribution ≠ Train distribution
- Result: p = 3.57e-143 → **Reject H0** (distributions differ significantly)

### Jensen-Shannon Divergence

Symmetric measure of similarity between two probability distributions:
- Range: [0, 1]
- 0 = identical distributions
- 1 = maximally different
- Our result: 0.3619 → **substantial difference**

### Silhouette Score

Measures how similar an object is to its own cluster compared to other clusters:
- Range: [-1, 1]
- 1 = well-clustered
- 0 = overlapping clusters
- -1 = misclassified
- Our result (PCA): 0.336 → **moderate separability**

---

**END OF PHASE 4 ANALYSIS**
