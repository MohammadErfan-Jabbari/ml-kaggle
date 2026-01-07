# Decision Tree - Experiment Paths

This document tracks all major decision branches explored during the competition.

## How to Read This Tree

- **→** indicates a decision/experiment
- **✅** path taken, succeeded
- **❌** path taken, failed/worse
- **⏳** path planned but not yet explored
- **🔄** currently in progress

---

## Root: Competition Start

```
Competition Start (2025-01-07)
│
├─→ Phase 0: Foundation Fixes
│   ├─→ H1: Fix NaN Label Masking ⏳
│   │   └─→ Implement MaskedBCEWithLogitsLoss
│   │
│   └─→ Implement Stratified CV ⏳
│       └─→ MultilabelStratifiedKFold + species
│
├─→ Phase 1: Baselines (after Phase 0)
│   ├─→ H0: MLP Baseline ⏳
│   │   ├─→ If AUC > 0.75: Focus on NN improvements
│   │   └─→ If AUC < 0.72: Consider GBM focus
│   │
│   └─→ H2: LightGBM Baseline ⏳
│       ├─→ If beats MLP: Focus on GBM + ensemble
│       └─→ If MLP wins: Focus on NN architectures
│
├─→ Phase 2: Species-Aware (after Phase 1)
│   ├─→ H3: Species Reweighting ⏳
│   │   └─→ Match test distribution
│   │
│   ├─→ H7: Species-Specific Models ⏳
│   │   └─→ Train 4 separate models
│   │
│   └─→ H8: Deterministic Rules for P. aeruginosa ⏳
│       └─→ Predict 1 for intrinsically resistant antibiotics
│
├─→ Phase 3: Architecture (parallel with Phase 2)
│   ├─→ H4: 1D CNN ⏳
│   │   └─→ Multi-scale kernels [5, 11, 21]
│   │
│   └─→ Attention MLP ⏳
│       └─→ Feature importance via attention
│
├─→ Phase 4: Semi-Supervised (after Phase 2-3)
│   ├─→ H5: Pseudo-Labeling ⏳
│   │   └─→ Focus on Amox/Clav (43% missing)
│   │
│   └─→ Consistency Regularization ⏳
│
└─→ Phase 5: Ensemble (final phase)
    ├─→ H6: LightGBM + CNN Blend ⏳
    │
    └─→ Stacking with Meta-Learner ⏳
```

---

## Detailed Branch Records

### Branch: Phase 0 - Foundation
**Started**: 2025-01-07
**Status**: 🔄 In Progress

| Decision | Outcome | Notes |
|----------|---------|-------|
| Fix NaN masking | ⏳ Pending | Required before any training |
| Stratified CV | ⏳ Pending | Required for valid evaluation |

---

### Branch: Phase 1 - Baselines
**Started**: Not yet
**Status**: ⏳ Pending Phase 0

| Decision | Outcome | Val AUC | LB AUC | Notes |
|----------|---------|---------|--------|-------|
| MLP Baseline | ⏳ | - | - | - |
| LightGBM | ⏳ | - | - | - |

---

## Path Summary

| Path ID | Description | Final AUC | Status |
|---------|-------------|-----------|--------|
| P1 | MLP → Fix bugs → Train | - | ⏳ |
| P2 | LightGBM baseline | - | ⏳ |
| P3 | MLP + LightGBM ensemble | - | ⏳ |

---

## Abandoned Paths

(None yet - just starting)

| Path | Why Abandoned | Learning |
|------|---------------|----------|
| - | - | - |

---

## Key Decision Points Log

| Date | Decision | Chosen Path | Rationale |
|------|----------|-------------|-----------|
| 2025-01-07 | Initial approach | Hypothesis-driven | Systematic exploration |
| 2025-01-07 | First model | MLP + LightGBM parallel | Compare NN vs GBM |
