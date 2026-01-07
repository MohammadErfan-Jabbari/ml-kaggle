# EDA Conclusions & Updated Modeling Strategy

> **Document Purpose**: This is the authoritative summary of all EDA findings. All agents should read this before making modeling decisions.
>
> **Date**: 2026-01-07
> **Based On**: 8 comprehensive EDA phases (45+ analyses)

---

## Executive Summary: The 5 Critical Truths

After 8 phases of comprehensive EDA, these are the non-negotiable facts that must drive every modeling decision:

### #1: Severe Species Distribution Shift (CRITICAL)
| Species | Train % | Test % | Shift |
|---------|---------|--------|-------|
| P. aeruginosa | **43.1%** | **3.0%** | **-40.1 pp** |
| K. pneumoniae | 27.9% | **50.8%** | +22.9 pp |
| E. coli | 16.6% | **26.9%** | +10.3 pp |
| P. mirabilis | 12.4% | 19.3% | +6.9 pp |

**Statistical Significance**: χ² p = 3.57e-143 (essentially infinite)
**Domain Classifier AUC**: 0.6677 (clearly distinguishable)

**Implication**: A model trained normally will overfit to P. aeruginosa patterns and catastrophically fail on test (where K. pneumoniae dominates).

---

### #2: Intrinsic Resistance = Free Predictions

| Species | Antibiotic | Resistance Rate | Action |
|---------|------------|-----------------|--------|
| P. aeruginosa | Ampicillin | 100% | **Predict 1.0** |
| P. aeruginosa | Amox/Clav | 100%* | **Predict 1.0** |
| P. aeruginosa | Ertapenem | 100% | **Predict 1.0** |
| P. aeruginosa | Cefotaxime | 100% | **Predict 1.0** |
| P. aeruginosa | Cefuroxime | 100% | **Predict 1.0** |
| P. mirabilis | Imipenem | 97.2% | **Predict 1.0** |

*Only 54 labeled samples for P. aeruginosa

**Implication**: 18.8% of predictions can be perfectly accurate with simple rules. No ML needed for these cases.

---

### #3: Systematic Label Missingness

| Antibiotic | Missing % | Primary Species Affected |
|------------|-----------|-------------------------|
| Amox/Clav | **42.8%** | P. aeruginosa: 96.3% missing |
| Imipenem | 3.3% | P. mirabilis: 14.2% missing |
| Fluoroquinolones | ~2.6% | E. coli: ~5% missing |

**Implication**: Missingness is biologically meaningful (not random), species-specific, and must be handled with masked loss.

---

### #4: Extreme Feature Sparsity

| Metric | Value |
|--------|-------|
| Zero values | **93.3%** |
| Constant features (var≈0) | **365 (6.1%)** |
| Near-constant (var<1e-5) | **400 (6.7%)** |
| Total removable | **~765 features (12.8%)** |

**Implication**: Tree-based models (LightGBM) will significantly outperform dense neural networks on this data.

---

### #5: Antibiotic Correlation Structure

| Pair | Correlation | Interpretation |
|------|-------------|----------------|
| **Levofloxacin ↔ Ciprofloxacin** | **0.925** | Same class (fluoroquinolones) |
| **Ertapenem ↔ Cefotaxime** | **0.813** | Cross-class β-lactam synergy |
| **Imipenem ↔ Ertapenem** | **0.772** | Same class (carbapenems) |

**Implication**: Multi-task learning with shared representations is justified. Task grouping by drug class should improve performance.

---

## Revised Modeling Strategy

### Architecture Decision Tree

```
START
  │
  ├─> For P. aeruginosa samples:
  │     ├─> Predict 1.0 for: Ampicillin, Ertapenem, Cefotaxime, Cefuroxime, Amox/Clav
  │     └─> Use ML for: Levofloxacin, Ciprofloxacin, Imipenem
  │
  ├─> For P. mirabilis samples:
  │     ├─> Predict 1.0 for: Imipenem
  │     └─> Use ML for: All other antibiotics
  │
  └─> For E. coli & K. pneumoniae:
        └─> Use ML for: All antibiotics
```

### Recommended Model Choice

**Primary Recommendation: LightGBM**

| Factor | LightGBM | Neural Network | Winner |
|--------|----------|----------------|--------|
| Handles 93% sparsity | ✓ Native | ✗ Needs special handling | **LightGBM** |
| Small data (3360 samples) | ✓ Excellent | ✗ Prone to overfit | **LightGBM** |
| Missing labels | ✓ Native | ✗ Custom loss needed | **LightGBM** |
| Feature importance | ✓ Built-in | ✗ Needs SHAP/Permutation | **LightGBM** |
| Training speed | ✓ Fast | ✗ Slow | **LightGBM** |
| Non-linear boundaries | ✓ Excellent | ✓ Excellent | Tie |

**Verdict**: Start with LightGBM. Only consider NN if LightGBM plateaus.

---

## Mandatory Implementation Requirements

### 1. Species-Stratified Cross-Validation (NON-NEGOTIABLE)

```python
from sklearn.model_selection import StratifiedKFold

# MUST stratify by species_id
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, species_id):
    # Each fold has proportional species representation
    # More similar to test distribution
```

**Why**: Random CV will give inflated scores (0.85-0.88) that don't generalize to test.

---

### 2. Sample Reweighting (NON-NEGOTIABLE)

```python
# Counteract P. aeruginosa overrepresentation
sample_weights = np.where(
    species_id == 3,  # P. aeruginosa
    0.3,              # Downweight by 70%
    1.0               # Normal weight for others
)
```

**Why**: Prevents model from optimizing for P. aeruginosa at expense of K. pneumoniae (which is 51% of test).

---

### 3. Masked Loss for Missing Labels (NON-NEGOTIABLE)

```python
# For LightGBM: Use custom objective or handle in dataset
# For Neural Network:
class MaskedBCEWithLogitsLoss(nn.Module):
    def forward(self, predictions, targets):
        mask = targets.notna()
        loss = F.binary_cross_entropy_with_logits(
            predictions[mask],
            targets[mask].float()
        )
        return loss.mean()
```

**Why**: Amox/Clav has 42.8% missing labels (96.3% for P. aeruginosa). Standard loss will fail.

---

### 4. Per-Species Metric Tracking (NON-NEGOTIABLE)

```python
# Track during validation
metrics = {
    'overall_auc': mean_auc,
    'e_coli_auc': e_coli_auc,
    'k_pneumoniae_auc': k_pneumoniae_auc,  # Most important for test!
    'p_mirabilis_auc': p_mirabilis_auc,
    'p_aeruginosa_auc': p_aeruginosa_auc
}
```

**Why**: K. pneumoniae is 51% of test but only 28% of train. This is the canary in the coal mine.

---

## Feature Engineering Pipeline

### Step 1: Remove Constant Features
```python
# Remove ~765 constant/near-constant features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=1e-5)
X_filtered = selector.fit_transform(X)
```

### Step 2: Add Species Embedding (if using NN)
```python
# One-hot encode species_id
species_embedding = nn.Embedding(4, 8)  # 4 species, 8-dim embedding
```

### Step 3: Consider Species-Specific Biomarkers
```python
# Top discriminating features from EDA
biomarker_features = [3165, 1999, 3290, 3435, 3434]  # From Phase 4
# Feature 3165 → P. mirabilis marker
# Feature 3290 → K. pneumoniae marker
# Features 3435, 3434 → P. aeruginosa markers
```

---

## Updated Hypothesis Priority Queue

### H1: Fix NaN Label Handling (PREREQUISITE)
- Status: ⏳ Pending
- Priority: 0 (blocks everything)
- Implementation: MaskedBCEWithLogitsLoss

### H2: LightGBM Baseline with Proper Validation
- Status: ⏳ Pending
- Priority: 1
- Configuration:
  - Species-stratified 5-fold CV
  - Sample weights (0.3x for P. aeruginosa)
  - Masked loss for missing labels
  - Remove constant features

### H3: Hybrid Rule + ML Architecture
- Status: ⏳ Pending
- Priority: 2
- Expected gain: +0.05-0.10 AUC from intrinsic resistance rules

### H4: Multi-Task Learning (LightGBM or NN)
- Status: ⏳ Pending
- Priority: 3
- Focus on correlated pairs (Levo/Cipro, Imi/Ert, Cefotaxime/Ert)

### H5: Species-Specific Models
- Status: ⏳ Pending
- Priority: 4
- Train 4 independent models, ensemble at inference

---

## Success Metrics & Validation

### What Success Looks Like

| Metric | Target | Warning Threshold |
|--------|--------|-------------------|
| Overall Mean AUC | > 0.85 | < 0.80 |
| K. pneumoniae AUC | > 0.82 | < 0.75 (majority in test!) |
| E. coli AUC | > 0.85 | < 0.80 |
| Per-species variance | < 0.05 | > 0.10 (overfitting warning) |

### Red Flags to Watch

1. **Validation AUC >> K. pneumoniae AUC**: Model overfitting to P. aeruginosa
2. **All predictions > 0.8 for E. coli**: Model predicting high resistance universally (should be ~40%)
3. **Feature importance dominated by species biomarkers**: Model not learning resistance patterns
4. **P. aeruginosa loss << other species**: Distribution shift not addressed

---

## Updated Approach Plan

### Phase 0: Foundation (Current)
- ✅ EDA Complete
- ⏳ Fix NaN label handling
- ⏳ Implement species-stratified CV
- ⏳ Set up sample weighting

### Phase 1: Baselines (Next)
- ⏳ LightGBM baseline (expected AUC: 0.80-0.85)
- ⏳ Add intrinsic resistance rules (expected +0.05-0.10)
- ⏳ First Kaggle submission

### Phase 2: Iteration
- ⏳ Multi-task learning (if baseline > 0.82)
- ⏳ Feature selection/engineering
- ⏳ Species-specific models

### Phase 3: Advanced
- ⏳ Ensembling
- ⏳ Pseudo-labeling for Amox/Clav
- ⏳ Neural network exploration (if LightGBM plateaus)

---

## Code Checklist

Before training any model, ensure:

- [ ] `species_id` is used for stratified CV
- [ ] Sample weights applied (0.3x for P. aeruginosa)
- [ ] Masked loss implemented for NaN labels
- [ ] Constant features removed (~765 features)
- [ ] Intrinsic resistance rules applied before ML prediction
- [ ] Per-species metrics tracked during validation
- [ ] K. pneumoniae AUC specifically monitored

---

## Key Insights by Phase

| Phase | Critical Finding | Action Required |
|-------|------------------|-----------------|
| 1: Data Profiling | 42.8% Amox/Clav missing | Masked loss |
| 2: Features | 12.8% constant features | Remove them |
| 3: Targets | Resistance 29.9%-88.2% | Class weighting |
| 4: Species | P. aeruginosa 43%→3% | Reweight samples |
| 5: Dim Reduction | 433 PCs for 90% variance | Use full features |
| 6: Train-Test Shift | Domain classifier AUC 0.67 | Species-stratified CV |
| 7: Feature-Target | Feature importance varies by species | Species-aware modeling |
| 8: Biological | 18.8% intrinsically resistant | Rule-based predictions |

---

## References

### Insight Documents
- `knowledge/insights/eda_phase1_data_profiling.md` - Data quality, missingness, species shift
- `knowledge/insights/eda_phase2_feature_analysis.md` - Sparsity, variance, correlations
- `knowledge/insights/eda_phase3_target_analysis.md` - Class balance, target correlations
- `knowledge/insights/eda_phase4_species_analysis.md` - Species shift (CRITICAL), intrinsic resistance
- `knowledge/insights/eda_phase5_dimensionality.md` - PCA, clustering, separability
- `knowledge/insights/eda_phase6_train_test_distribution.md` - Covariate shift detection
- `knowledge/insights/eda_phase7_feature_target.md` - Feature importance, biomarkers
- `knowledge/insights/eda_phase8_biological_analysis.md` - Antibiotic classes, MDR, rules

### Figure Locations
All figures: `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase*/`

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-07 | Choose LightGBM over NN as baseline | 93% sparsity, small dataset (3360 samples) |
| 2026-01-07 | Mandatory species-stratified CV | χ² p = 3.57e-143 for distribution shift |
| 2026-01-07 | Implement intrinsic resistance rules | 18.8% of predictions can be perfect |
| 2026-01-07 | Downweight P. aeruginosa 3x | 43% train vs 3% test distribution |
| 2026-01-07 | Use masked loss for missing labels | 42.8% missing for Amox/Clav, systematic |

---

*This document is the single source of truth for modeling decisions. All agents must read and adhere to these conclusions before implementing any changes.*
