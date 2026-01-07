# Session State

> **Last Updated**: 2026-01-07 (Session 3 - Course Methods Analysis)
> **Session**: 3 (Course Methods Analysis) ✅ COMPLETE

---

## 🚨 READ THIS FIRST 🚨

**Before making ANY modeling decisions, you MUST read:**
1. `knowledge/EDA_CONCLUSIONS_STRATEGY.md` - 5 Critical Truths from EDA
2. `knowledge/COURSE_METHODS_REFERENCE.md` - Course methods mapped to competition

**Quick summary if you don't read them:**

**From EDA:**
1. Use **LightGBM** (not NN) - 93% sparsity, small data
2. **Species-stratified CV** is MANDATORY - P. aeruginosa: 43%→3% distribution shift
3. **Sample weights**: 0.3x for P. aeruginosa
4. **Rule-based predictions**: 18.8% can be perfect (intrinsic resistance)
5. **Masked loss** for missing labels (Amox/Clav: 42.8% missing)

**From Course:**
1. **Leakage-safe pipelines**: Fit feature extractors only within CV folds
2. **PLS/PCA > raw features**: Supervised dimensionality reduction
3. **Semi-supervised**: K-Means + PLS for missing labels
4. **One-Class SVM**: Detect OOD samples from species shift

---

## Current Status

| Metric | Value |
|--------|-------|
| Best Validation AUC | Not established |
| Best Public LB | Not submitted |
| Current Phase | Phase 1 - Baselines |
| Active Hypothesis | H2 (LightGBM Baseline) |
| EDA Status | ✅ COMPLETE (8 phases, 45+ analyses) |
| Course Methods Analysis | ✅ COMPLETE (13 notebooks analyzed) |

---

## EDA Summary (Session 2)

### Comprehensive Analysis Completed
- ✅ Phase 1: Data Profiling & Quality Assessment
- ✅ Phase 2: Feature Space Analysis (6000 MALDI features)
- ✅ Phase 3: Target Analysis (8 antibiotics)
- ✅ Phase 4: Species Analysis (distribution shift)
- ✅ Phase 5: Dimensionality Reduction & Clustering
- ✅ Phase 6: Train-Test Distribution Analysis
- ✅ Phase 7: Feature-Target Relationships
- ✅ Phase 8: Biological/Domain-Specific Analysis

### Critical Findings (5 Truths)

**#1: Severe Species Distribution Shift** (χ² p = 3.57e-143)
- P. aeruginosa: 43.1% → 3.0% (-40.1 pp)
- K. pneumoniae: 27.9% → 50.8% (+22.9 pp)
- **Implication**: Must use species-stratified CV + sample reweighting

**#2: Intrinsic Resistance = Free Predictions**
- P. aeruginosa: 100% resistant to 5/8 antibiotics
- P. mirabilis: 97.2% resistant to Imipenem
- **Implication**: 18.8% of predictions can be rule-based (perfect accuracy)

**#3: Systematic Label Missingness**
- Amox/Clav: 42.8% missing (96.3% for P. aeruginosa)
- **Implication**: Must use masked loss

**#4: Extreme Feature Sparsity**
- 93.3% zero values
- 12.8% constant/near-constant features
- **Implication**: LightGBM favored over NN

**#5: Antibiotic Correlation Structure**
- Levo ↔ Cipro: r=0.925 (same class)
- Ertapenem ↔ Cefotaxime: r=0.813
- **Implication**: Multi-task learning justified

### Deliverables Created
- 45+ figures in `outputs/eda/phase*/`
- 8 insight documents in `knowledge/insights/`
- 8 analysis scripts in `scripts/eda/`
- **Strategic summary**: `knowledge/EDA_CONCLUSIONS_STRATEGY.md` ⭐ READ THIS FIRST

---

## Updated Strategy (Based on EDA)

### Model Choice Decision: LightGBM
**Rationale**:
- 93% sparsity (handles natively)
- 3360 samples (small for NN)
- Missing label handling (native)
- Training speed (fast)
- Feature importance (built-in)

### Architecture: Hybrid Rule + ML
```
For P. aeruginosa samples:
  → Predict 1.0 for: Ampicillin, Ertapenem, Cefotaxime, Cefuroxime, Amox/Clav
  → Use ML for: Levofloxacin, Ciprofloxacin, Imipenem

For P. mirabilis samples:
  → Predict 1.0 for: Imipenem
  → Use ML for: All other antibiotics

For E. coli & K. pneumoniae:
  → Use ML for: All antibiotics
```

### Mandatory Requirements (NON-NEGOTIABLE)
1. ✅ Species-stratified cross-validation
2. ✅ Sample reweighting (0.3x for P. aeruginosa)
3. ✅ Masked loss for missing labels
4. ✅ Per-species metric tracking
5. ✅ Remove constant features (~765)

---

## Course Methods Summary (Session 3)

### 13 Notebooks Analyzed
**Key Deliverable**: `knowledge/COURSE_METHODS_REFERENCE.md`

### Methods Mapped to Competition
| Challenge | Course Solution |
|-----------|-----------------|
| Missing labels (42.8%) | K-Means clustering, PLS regression, kernel similarity |
| 6000 MALDI features | PCA → PLS/CCA → ARD (feature selection) |
| Species shift | One-Class SVM, Bayesian uncertainty, sample reweighting |
| Small data (3360) | LightGBM/GBM, Bayesian priors, ensemble methods |

### Course → AMR Applications
- **Semi-supervised**: K-Means + PLS for missing label imputation
- **Feature extraction**: PLS/CCA (supervised) > PCA for finding resistance peaks
- **Novelty detection**: One-Class SVM for OOD detection (species shift)
- **Ensembles**: Bagging with max_features sampling for variance reduction
- **Bayesian**: Uncertainty quantification for low-confidence predictions

### Critical Emphasis from Course
**Leakage-safe pipelines**: Fit feature extractors (PCA/PLS/CCA) only within CV folds, not on entire dataset. Essential for small data (3360 samples).

**Reference**: `knowledge/COURSE_METHODS_REFERENCE.md` ⭐

---

## What to Do Next

### Immediate Priority (Start Here)

1. **Implement core infrastructure** (`src/utils/`)
   - [ ] `MaskedBCEWithLogitsLoss` for missing labels
   - [ ] `get_species_stratified_cv()` for proper splitting
   - [ ] `get_sample_weights()` for P. aeruginosa downweighting
   - [ ] `intrinsic_resistance_rules()` for rule-based predictions

2. **LightGBM baseline** (`src/models/lightgbm_baseline.py`)
   - [ ] Remove constant features
   - [ ] Train 8 separate LightGBM models (one per antibiotic)
   - [ ] Use species-stratified 5-fold CV
   - [ ] Apply sample weights
   - [ ] Track per-species and per-antibiotic AUC

3. **Hybrid Rule + ML inference** (`src/inference/predict.py`)
   - [ ] Apply intrinsic resistance rules first
   - [ ] Use LightGBM for remaining predictions
   - [ ] Generate submission.csv

4. **First Kaggle submission**
   - [ ] Submit baseline
   - [ ] Record LB score as benchmark

### Then

5. **Iterate based on baseline performance**
   - [ ] If AUC < 0.80: Debug, check per-species metrics
   - [ ] If AUC 0.80-0.85: Add multi-task learning
   - [ ] If AUC > 0.85: Try ensembling, species-specific models

---

## Blockers

| Blocker | Impact | Resolution |
|---------|--------|------------|
| Infrastructure code | Can't train properly | Implement utils (see above) |
| NaN label handling | Training will fail | Implement masked loss |
| Species shift | Validation not representative | Species-stratified CV |

---

## Hypothesis Queue (Updated)

| Priority | Hypothesis | Status | Expected Gain |
|----------|------------|--------|---------------|
| 0 | H1: Fix NaN Label Handling | ⏳ Pending | Prerequisite |
| 1 | H2: LightGBM Baseline | ⏳ Pending | Baseline AUC: 0.80-0.85 |
| 2 | H3: Hybrid Rule + ML | ⏳ Pending | +0.05-0.10 AUC |
| 3 | H4: Multi-Task Learning | ⏳ Pending | +0.01-0.03 AUC |
| 4 | H5: Species-Specific Models | ⏳ Pending | +0.02-0.05 AUC |

---

## Quick Commands for Next Session

```bash
# Run LightGBM baseline (after implementing utils)
uv run python src/models/lightgbm_baseline.py

# Generate submission
uv run python src/inference/predict.py --checkpoint outputs/models/lgb_baseline

# Submit to Kaggle
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/submissions/submission.csv -m "LightGBM baseline + species stratification"
```

---

## Key Documents to Read

**Mandatory before implementing**:
1. `knowledge/EDA_CONCLUSIONS_STRATEGY.md` - ⭐ Comprehensive strategy from EDA
2. `knowledge/insights/eda_phase4_species_analysis.md` - Species shift details
3. `knowledge/insights/eda_phase8_biological_analysis.md` - Intrinsic resistance rules

**For reference**:
4. `knowledge/insights/eda_phase2_feature_analysis.md` - Feature sparsity
5. `knowledge/insights/eda_phase7_feature_target.md` - Feature importance

---

## Success Metrics

| Metric | Target | Warning |
|--------|--------|---------|
| Overall Mean AUC | > 0.85 | < 0.80 |
| K. pneumoniae AUC | > 0.82 | < 0.75 |
| E. coli AUC | > 0.85 | < 0.80 |
| Per-species variance | < 0.05 | > 0.10 |

---

## Notes for Next Session

**CRITICAL**: Read `knowledge/EDA_CONCLUSIONS_STRATEGY.md` before implementing anything.

**Key changes from EDA**:
- Use LightGBM instead of MLP as baseline
- MUST implement species-stratified CV
- MUST downweight P. aeruginosa samples (0.3x)
- Apply intrinsic resistance rules before ML prediction
- Track K. pneumoniae AUC specifically (51% of test!)

**Red flags**:
- Validation AUC >> K. pneumoniae AUC → overfitting to P. aeruginosa
- All predictions > 0.8 for E. coli → not learning patterns
- P. aeruginosa loss << other species → distribution shift not addressed
