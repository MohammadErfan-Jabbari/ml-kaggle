# Hypothesis Tracker - AMR Prediction Competition

## Current State
**Date**: 2025-01-07 (Session 1 Complete)
**Best Validation AUC**: Not yet established (baseline not run)
**Best Public LB**: Not yet submitted
**Next Session**: Implement H1 (Fix NaN Labels) → Run baseline

---

## Hypothesis Queue

### H0: Baseline MLP (Current Focus)
**Status**: 🔄 Ready to run (requires bug fix first)

**Hypothesis**: The existing MLP baseline with species embedding can achieve reasonable AUC (>0.70) on validation set.

**Experiment**:
1. Fix NaN label handling in loss function
2. Implement proper stratified CV
3. Train and evaluate baseline

**Expected Outcome**: AUC ~0.70-0.75

**Actual Outcome**: TBD

---

### H1: Fix NaN Label Masking
**Status**: ⏳ Pending (prerequisite for all experiments)

**Hypothesis**: Properly masking NaN labels in BCE loss will enable training without NaN gradients.

**Change**: Implement `MaskedBCEWithLogitsLoss` in training loop

**Risk**: Low - this is a bug fix, not a modeling change

---

### H2: LightGBM Baseline
**Status**: ⏳ Pending

**Hypothesis**: LightGBM will outperform MLP on this small dataset (3360 samples, 6000 features) based on literature showing gradient boosting wins on tabular data with <5000 samples.

**Experiment**:
1. Train separate LightGBM per antibiotic
2. Handle NaN labels by filtering
3. Use StratifiedKFold with species stratification

**Expected Outcome**: AUC +0.02-0.05 over MLP baseline

---

### H3: Species-Weighted Training
**Status**: ⏳ Pending

**Hypothesis**: Upweighting non-P. aeruginosa samples during training will improve generalization to test set (which has only 3% P. aeruginosa vs 43% in train).

**Experiment**: Apply sample weights inversely proportional to species frequency in train OR proportional to frequency in test distribution.

**Expected Outcome**: Better calibration on test set

---

### H4: 1D CNN Architecture
**Status**: ⏳ Pending

**Hypothesis**: 1D CNN will capture local spectral patterns (adjacent m/z features) better than MLP, which treats features independently.

**Architecture**:
- Multi-scale kernels: [5, 11, 21]
- BatchNorm + ReLU + MaxPool
- Species embedding concatenated after conv layers

**Expected Outcome**: AUC +0.03-0.08 over MLP

---

### H5: Pseudo-Labeling for Missing Labels
**Status**: ⏳ Pending (after H1-H4)

**Hypothesis**: Using pseudo-labels for the 1439 samples with missing Amox/Clav labels will improve that antibiotic's AUC.

**Approach**:
1. Train model on labeled data
2. Predict Amox/Clav for unlabeled samples
3. Add high-confidence predictions (>0.9) as pseudo-labels
4. Retrain

**Expected Outcome**: Amox/Clav AUC improvement

---

### H6: Ensemble (LightGBM + CNN)
**Status**: ⏳ Pending (after H2, H4)

**Hypothesis**: Averaging predictions from LightGBM and CNN will outperform either alone due to model diversity.

**Blend**: `0.5 * LightGBM + 0.5 * CNN` or optimize weights on validation

**Expected Outcome**: AUC +0.02-0.03 over best single model

---

### H7: Species-Specific Models
**Status**: ⏳ Pending

**Hypothesis**: Training separate models per species will capture species-specific resistance patterns better than a single model with species embedding.

**Challenge**: P. mirabilis has only 415 samples - may need regularization

**Expected Outcome**: Potentially large gains for E. coli and K. pneumoniae

---

### H8: Deterministic Rules for P. aeruginosa
**Status**: ⏳ Pending

**Hypothesis**: For P. aeruginosa (species_id=3), predicting resistance=1 for Ampicillin, Ertapenem, Cefotaxime, Cefuroxime will be nearly perfect based on 100% resistance rate in training data.

**Risk**: If test has different P. aeruginosa resistance patterns, this will hurt. But only 3% of test is P. aeruginosa.

**Expected Outcome**: Perfect accuracy for these 4 antibiotics on P. aeruginosa samples

---

## Completed Hypotheses

(None yet - competition just started)

---

## Experiment Log

| Date | Hypothesis | Validation AUC | Public LB | Notes |
|------|------------|----------------|-----------|-------|
| - | - | - | - | - |

---

## Key Learnings

1. Species distribution shift is the biggest challenge (P. aeruginosa 43% → 3%)
2. P. aeruginosa has intrinsic resistance to 5/8 antibiotics
3. Fluoroquinolones (Levo/Cipro) are highly correlated (r=0.92)
4. Amox/Clav has 43% missing labels
5. Data is 93.4% sparse (zeros)

---

## Next Session Checklist

1. [ ] Fix NaN label masking (H1)
2. [ ] Implement proper stratified CV
3. [ ] Run baseline and establish benchmark
4. [ ] Try LightGBM (H2)
5. [ ] Submit first prediction to get public LB score
