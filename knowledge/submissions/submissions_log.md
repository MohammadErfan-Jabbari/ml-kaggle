# Submission Log

This file tracks all Kaggle submissions with their local and leaderboard performance.

---

## Submission #1: Baseline
**Date**: 2026-01-07 17:28
**File**: `sub_lgb_baseline_20260107_1430.csv`
**Model**: LGB
**Description**: baseline

### Local OOF Performance
| Metric | Value |
|--------|-------|
| Mean AUC | 0.8978 |
| Ampicillin | 0.9282 |
| Levofloxacin | 0.8461 |
| Ciprofloxacin | 0.8454 |
| Imipenem | 0.9911 |
| Amoxicillin_Clavulanic_acid | 0.7078 |
| Ertapenem | 0.9862 |
| Cefotaxime | 0.9349 |
| Cefuroxime | 0.9426 |

### Configuration
```python
{
  "n_estimators": 100,
  "learning_rate": 0.1,
  "num_leaves": 31,
  "min_child_samples": 20,
  "subsample": 0.8,
  "colsample_bytree": 0.8
}
```

### Kaggle Leaderboard
- **Public LB**: 0.8324

- **Private LB**: _pending final results_

### Notes
Baseline LightGBM with species-stratified 5-fold CV, no sample reweighting, no intrinsic rules. Renamed from submission.csv.

---

## Submission #2: Reweight_Intrinsic
**Date**: 2026-01-07 17:53
**File**: `sub_lgb_reweight_intrinsic_20260107_1753.csv`
**Model**: LGB
**Description**: reweight_intrinsic

### Local OOF Performance
| Metric | Value |
|--------|-------|
| Mean AUC | 0.8978 |
| Ampicillin | 0.9282 |
| Levofloxacin | 0.8461 |
| Ciprofloxacin | 0.8454 |
| Imipenem | 0.9911 |
| Amoxicillin_Clavulanic_acid | 0.7078 |
| Ertapenem | 0.9862 |
| Cefotaxime | 0.9349 |
| Cefuroxime | 0.9426 |

### Configuration
```python
{
  "n_estimators": 100,
  "learning_rate": 0.1,
  "num_leaves": 31,
  "min_child_samples": 20,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "sample_weight": "0.3x for P.aeruginosa",
  "intrinsic_rules": true
}
```

### Kaggle Leaderboard
- **Public LB**: 0.8328

- **Private LB**: _pending final results_

### Notes
LightGBM with sample reweighting (0.3x P.aeruginosa) + intrinsic resistance rules

---

## Session 4 Analysis

### Why the Gap? (OOF 0.8978 vs LB 0.8328 = -6.5%)

Per-species OOF metrics revealed the problem:

| Species | OOF AUC | % of Train | % of Test | Issue |
|---------|---------|------------|-----------|-------|
| P. aeruginosa | nan (constant) | 43.1% | 3.0% | Model overfits to this |
| **K. pneumoniae** | **0.7313** | 27.9% | **50.8%** | **Majority of test - model fails** |
| E. coli | 0.7002 | 16.6% | 26.9% | Underrepresented in train |
| P. mirabilis | 0.7023 | 12.4% | 19.3% | Underrepresented in train |

**Conclusion**: The high overall OOF (0.8978) is inflated by P. aeruginosa (which has trivial constant predictions). The model actually performs poorly on K. pneumoniae (0.7313), which is 51% of the test set.

### Why Sample Reweighting Failed

- Reweighting changes loss weights but the model still learns the same features
- P. aeruginosa patterns still dominate because they're 43% of the data
- Need to train *separate* models per species to learn species-specific patterns

### Next Step: Species-Specific Models

Train 4 sets of LightGBM models (one per species). At inference, route each test sample to its species-specific model.

---

## Submission #3: Mega-Blend Rank Averaging
**Date**: 2026-01-07 20:57
**File**: `sub_mega_blend_rank_avg_20260107_2057.csv`
**Model**: Ensemble (LGB + PLS + Species-Global Blend)
**Description**: Mega-blend with rank averaging

### Local OOF Performance
| Metric | Value |
|--------|-------|
| Mean AUC | 0.9040 |
| K.pneumoniae AUC | 0.7728 |

### Configuration
```python
{
  "base_models": ["PLS-LGB", "Species-Global-Blend", "Tuned-LGB"],
  "blend_method": "rank_averaging",
  "species_weights": {"P.aeruginosa": 0.05, "K.pneumoniae": 3.0, "E.coli": 1.5, "P.mirabilis": 1.5},
  "lgb_params": {"n_estimators": 350, "num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 25}
}
```

### Kaggle Leaderboard
- **Public LB**: 0.83862

- **Private LB**: _pending final results_

### Notes
Session 5 comprehensive experimentation: 15+ methods tested across 3 rounds. Best approach was rank-averaging ensemble of diverse models. K.pn OOF AUC improved from 0.7313 to 0.7728 (+4.15%), but LB only improved from 0.8328 to 0.83862 (+0.7%).

---

