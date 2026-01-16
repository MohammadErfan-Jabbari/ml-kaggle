# Hypothesis Tracker - Condensed

**Best LB**: 0.83862 | **Metric**: Val Mean AUC | **Date**: 2026-01-08

## Active Hypothesis: Miracle Blend v2

**Run**: `uv run python experiments/run_miracle_v2.py`

Builds 17 diverse models + rank averaging ensemble.

---

## Completed Hypotheses

| Hypothesis | Result | LB |
|------------|--------|-----|
| LightGBM Baseline | Works | 0.8324 |
| Species Reweighting | Minimal | 0.8328 |
| Mega-blend (3 models) | **BEST** | **0.83862** |
| Stacking | OVERFIT | 0.8269 |
| K.pn AUC optimization | HURTS LB | 0.82451 |
| PLS(n=20) | Good locally | - |
| Per-species models | No improvement | - |

---

## Key Learnings

1. **Metric**: Use Val Mean AUC (not K.pn AUC)
2. **Stacking**: Overfits 12.8% - avoid
3. **Rank averaging**: Proven robust
4. **Model diversity**: LGB + XGB + CatBoost + MLP
5. **Validation split**: Must match test distribution

---

## Don't Try Again

- Stacking/meta-learners
- K.pn AUC as selection metric
- Complex feature engineering
- Single model submissions
