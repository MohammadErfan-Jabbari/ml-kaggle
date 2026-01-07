# Competition Approach Plan

## Executive Summary

**Competition**: Antimicrobial Resistance Prediction from MALDI-TOF
**Task**: Multi-label, semi-supervised classification predicting resistance (0/1) to 8 antibiotics
**Metric**: Mean AUC across 8 antibiotics
**Key Challenge**: Species distribution shift (P. aeruginosa 43%→3%), partially labeled data

---

## Phase 0: Foundation Fixes (Immediate)

### P0.1: Fix NaN Label Handling
**Priority**: CRITICAL (blocks all training)
**Location**: `src/training/train.py`

Current code computes loss on NaN labels, causing NaN gradients.

```python
# Fix: Implement masked BCE loss
class MaskedBCEWithLogitsLoss(nn.Module):
    def forward(self, logits, targets):
        mask = ~torch.isnan(targets)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        targets_clean = torch.where(mask, targets, torch.zeros_like(targets))
        loss = F.binary_cross_entropy_with_logits(logits, targets_clean, reduction='none')
        return (loss * mask.float()).sum() / mask.float().sum()
```

### P0.2: Implement Stratified Cross-Validation
**Priority**: HIGH
**Location**: `src/training/train.py`

Current 80/20 random split is inadequate. Implement:
1. Multi-label stratified K-fold (pip install iterative-stratification)
2. Include species in stratification
3. 5-fold CV for robust validation

---

## Phase 1: Establish Baselines (Week 1)

### P1.1: MLP Baseline (Current Architecture)
**Goal**: Establish baseline AUC with fixed training
**Expected AUC**: 0.70-0.75

Steps:
1. Apply P0 fixes
2. Run with current config
3. Log per-antibiotic AUC
4. Submit to get LB score

### P1.2: LightGBM Baseline
**Goal**: Test gradient boosting on this data
**Expected AUC**: 0.72-0.78

Literature suggests GBM often beats neural nets on:
- Small datasets (<5000 samples) ✓
- High-dimensional features (6000) ✓
- Tabular data ✓

Implementation:
```python
for antibiotic in antibiotics:
    mask = ~y[antibiotic].isna()
    model = LGBMClassifier(class_weight='balanced', n_estimators=500)
    model.fit(X[mask], y[antibiotic][mask])
```

### P1.3: Compare and Choose
- If LightGBM wins → focus on GBM optimization
- If MLP wins → focus on NN architectures
- If similar → plan ensemble

---

## Phase 2: Species-Aware Modeling (Week 2)

### P2.1: Address Distribution Shift
**Critical Insight**: Test set has only 3% P. aeruginosa (vs 43% train)

Options:
1. **Reweight samples**: Upweight E. coli, K. pneumoniae, P. mirabilis
2. **Subsample P. aeruginosa**: Match test distribution
3. **Species-specific models**: Train separate models, ensemble

Recommended: Start with reweighting, test species-specific if time permits.

### P2.2: Exploit P. aeruginosa Intrinsic Resistance
From data analysis, P. aeruginosa has 100% resistance to:
- Ampicillin, Ertapenem, Cefotaxime, Cefuroxime

For test samples with species_id=3:
- Set these 4 antibiotics to 1.0 (confident prediction)
- Only model Levofloxacin, Ciprofloxacin, Imipenem

### P2.3: Species Embedding vs Separate Models
Test both:
1. Current approach: Species embedding (32-dim) concatenated with MALDI features
2. Alternative: Train 4 separate models, one per species

Hypothesis: Separate models may capture species-specific resistance patterns better.

---

## Phase 3: Architecture Improvements (Week 3)

### P3.1: 1D CNN for Spectral Data
MALDI features are binned spectra with local structure. 1D CNN can capture:
- Peak patterns
- Adjacent feature correlations
- Multi-scale patterns (different kernel sizes)

Architecture:
```
Input (6000) → Conv1D(5) + Conv1D(11) + Conv1D(21) → Concat → Conv blocks → FC → 8 outputs
```

### P3.2: Attention Mechanism
Add feature attention to identify which m/z regions matter:
```
x → Attention(x) → Weighted features → MLP → outputs
```

Benefits:
- Interpretability
- Focus on discriminative regions (literature: 2000-7000 Da most important)

### P3.3: Label Correlation Exploitation
Strong correlations:
- Levo ↔ Cipro (r=0.92)
- Imipenem ↔ Ertapenem (r=0.77)
- Ertapenem ↔ Cefotaxime (r=0.81)

Options:
1. **Classifier chains**: Condition later predictions on earlier ones
2. **Multi-task with shared backbone**: Common features, separate heads
3. **Auxiliary losses**: Predict correlation patterns

---

## Phase 4: Semi-Supervised Learning (Week 4)

### P4.1: Pseudo-Labeling
For partially labeled samples (especially Amox/Clav with 43% missing):

1. Train model on labeled data
2. Predict on unlabeled samples
3. Add high-confidence (>0.9) pseudo-labels
4. Retrain with expanded dataset
5. Repeat 2-3 times with decreasing threshold

### P4.2: Consistency Regularization
Enforce prediction consistency under augmentation:
```python
pred_clean = model(x)
pred_noisy = model(x + noise)
consistency_loss = MSE(pred_clean, pred_noisy)
```

### P4.3: Self-Training
sklearn's SelfTrainingClassifier as wrapper around base model.

---

## Phase 5: Ensemble & Optimization (Week 5)

### P5.1: Model Diversity
Collect diverse models:
1. LightGBM with different parameters
2. XGBoost
3. CatBoost
4. MLP
5. 1D CNN
6. Attention MLP

### P5.2: Stacking
Level-0: K-fold OOF predictions from each model
Level-1: Train meta-model (LogisticRegression or simple NN) on OOF
Final: Average Level-1 predictions

### P5.3: Simple Blending
If stacking is overfit-prone:
```python
final = 0.4 * lgb + 0.3 * cnn + 0.3 * mlp
```

Optimize weights on validation set.

### P5.4: Test-Time Augmentation
For CNN:
- Predict with small noise variations
- Average predictions

---

## Phase 6: Final Tuning (Final Days)

### P6.1: Hyperparameter Search
Focus on:
- Learning rate
- Dropout
- Hidden layer sizes
- Regularization strength

### P6.2: Threshold Optimization
AUC is rank-based, but if submission expects probabilities:
- Calibrate using Platt scaling or isotonic regression

### P6.3: Final Ensemble
Select best 3-5 models based on CV score
Weight by validation performance
Verify no data leakage

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfit to train species distribution | Stratify by species, reweight samples |
| NaN label issues | Masked loss (P0.1) |
| Overfit to public LB (40%) | Trust CV more than LB |
| P. aeruginosa assumptions wrong | Only 3% of test, limited impact |
| Too many experiments, lose track | Use hypothesis tracker, MLflow |

---

## Success Metrics

| Milestone | Expected AUC | Timeline |
|-----------|--------------|----------|
| Baseline MLP | 0.70-0.75 | Day 2 |
| LightGBM | 0.72-0.78 | Day 3 |
| Best single model | 0.78-0.82 | Week 2 |
| Ensemble | 0.80-0.85 | Week 3 |
| Final submission | 0.82-0.87 | Week 4-5 |

---

## Files to Create/Modify

### Immediate (P0)
- [x] `knowledge/` - Knowledge management structure
- [ ] `src/training/train.py` - Fix NaN handling, add proper CV
- [ ] `src/utils/losses.py` - New file for MaskedBCEWithLogitsLoss

### Phase 1
- [ ] `src/models/lightgbm_baseline.py` - LightGBM wrapper
- [ ] `src/training/cv.py` - Cross-validation utilities

### Phase 2-3
- [ ] `src/models/cnn1d.py` - 1D CNN architecture
- [ ] `src/models/attention_mlp.py` - Attention mechanism
- [ ] `src/data/augmentation.py` - Spectral augmentation

### Phase 4-5
- [ ] `src/training/pseudo_label.py` - Pseudo-labeling pipeline
- [ ] `src/ensemble/blend.py` - Blending utilities
- [ ] `src/ensemble/stacking.py` - Stacking implementation

---

## Quick Reference Commands

```bash
# Train baseline
uv run python src/training/train.py

# Generate predictions
uv run python src/inference/predict.py --checkpoint outputs/models/best.pt

# Submit to Kaggle
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/submissions/submission.csv -m "Description"

# Start MLflow UI
uv run mlflow ui
```
