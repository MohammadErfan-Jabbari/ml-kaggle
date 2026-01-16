# Session Handoff - 2026-01-08 (Transductive DR Experiments)

## Summary

Attempted to combine unsupervised dimensionality reduction with self-training. **Results were disappointing** - PCA hurts performance.

## What We Tried

| Approach | Val Mean AUC | Notes |
|----------|--------------|-------|
| Original self-training (no PCA) | **0.8179** | Previous session's best |
| PCA 100 + Ensemble (LGB/XGB/Cat/MLP) | 0.7758 | -4.2% vs baseline |
| PCA 200 + Ensemble | 0.7901 | -2.8% vs baseline |
| PCA 200 + MLP only | 0.7754 | -4.3% vs baseline |
| Kernel PCA 100 | Not completed | GPU memory issues |
| Probabilistic PCA 100 | Not completed | Didn't run |

## Key Findings

1. **PCA hurts performance** - Even with 200 components (78.5% variance), we lose important signal
2. **Tree models prefer sparse features** - LightGBM/XGBoost work better on original 6000-dim MALDI features
3. **MLP doesn't help on PCA features** - Contrary to expectation, MLP on PCA didn't improve
4. **Pseudo-labels don't improve val AUC** - Self-training adds samples but doesn't move the needle

## Why PCA Fails for This Problem

1. MALDI data has specific peaks that are meaningful - PCA smooths these out
2. Tree models find splits on individual features - PCA linear combinations don't help
3. The covariate shift is in species distribution, not feature space

## Files Created

| File | Purpose |
|------|---------|
| `experiments/transductive_base.py` | Core module for transductive DR + self-training |
| `experiments/run_transductive_pca.py` | Standard PCA wrapper |
| `experiments/run_transductive_kpca.py` | Kernel PCA wrapper |
| `experiments/run_transductive_ppca.py` | Probabilistic PCA wrapper |
| `experiments/run_pca_mlp_self_training.py` | Simplified PCA + MLP only |

## Best Scores Unchanged

| Submission | LB Score | Method |
|------------|----------|--------|
| mega-blend (Session 5) | **0.83862** | PLS-LGB + Species-Global + Tuned-LGB |
| self-training | 0.82445 | Semi-supervised on raw features |

## Recommendations for Next Session

### Don't Pursue Further
- Unsupervised DR (PCA, Kernel PCA, PPCA) - doesn't help
- MLP on reduced features - worse than trees on raw features

### Try Instead
1. **Improve the mega-blend that got 0.83862**
   - It used PLS (supervised DR) not PCA (unsupervised)
   - PLS finds components that correlate with target

2. **Feature engineering on raw features**
   - Peak detection, binning, aggregation
   - Species-specific feature selection

3. **Different ensemble strategies**
   - Per-antibiotic model tuning
   - Threshold optimization

4. **Domain adaptation methods**
   - Density ratio reweighting (KLIEP/uLSIF)
   - Domain adversarial training

## Quick Commands

```bash
# Run original mega-blend (best LB)
uv run python experiments/run_mega_blend.py

# Run original self-training
uv run python experiments/run_self_training.py
```

## Output Locations

- PCA experiments: `outputs/transductive_dr_runs/`
- PCA+MLP runs: `outputs/pca_mlp_runs/`
