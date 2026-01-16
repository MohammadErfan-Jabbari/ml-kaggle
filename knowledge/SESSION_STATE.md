# Session State - AMR Competition

**Last Updated**: 2026-01-08 Session 17
**Best LB**: 0.83862 (mega-blend rank-avg)
**Primary Metric**: Val Mean AUC (avg across 8 antibiotics)

---

## Current Status: READY TO RUN MIRACLE BLEND

### Next Action
```bash
uv run python experiments/run_miracle_v2.py
```
This builds 17 diverse models + ensembles, evaluates with Val Mean AUC.

---

## Critical Lessons Learned

### 1. WRONG METRIC KILLED US (Session 14)
- We optimized K.pn AUC but LB scores Mean AUC
- Result: K.pn +5.6% but LB -1.6%
- **Fix**: All model selection now uses Val Mean AUC

### 2. STACKING MASSIVELY OVERFITS
- OOF Mean AUC = 0.9545 → LB = 0.8269 (gap: -12.8%)
- **Fix**: Use rank averaging only, no meta-learners

### 3. VALIDATION SPLIT IS ESSENTIAL
- Must match test species distribution
- Train: 2688, Val: 672 samples
- Load with: `load_validation_split()`

---

## Submission History

| Submission | Approach | LB Score |
|------------|----------|----------|
| Baseline | LightGBM | 0.8324 |
| + Reweight | Sample weights | 0.8328 |
| **Mega-blend** | **Rank-avg 3 models** | **0.83862** |
| Stacking | LGB meta-learner | 0.8269 (OVERFIT) |
| 5A2 Weighted | K.pn optimized | 0.82451 (WRONG METRIC) |

---

## What Works

| Method | Why |
|--------|-----|
| **Rank averaging** | Robust, proven best |
| Species weighting | Matches test distribution |
| Model diversity | LGB + XGB + CatBoost + MLP |
| Intrinsic rules | Free predictions (18.8%) |

## What Doesn't Work

| Method | Why |
|--------|-----|
| Stacking | 12.8% overfit gap |
| K.pn optimization | Hurts other antibiotics |
| Single models | Need ensemble diversity |

---

## Key Files

```
experiments/run_miracle_v2.py  # MAIN: Run this
experiments/run_mega_blend.py  # Previous best
src/data/dataset.py            # load_validation_split()
outputs/submissions/           # Generated submissions
```

---

## Quick Validation Check

```python
from src.data.dataset import load_validation_split
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
# Should be: Train: (2688, 6000), Val: (672, 6000)
```
