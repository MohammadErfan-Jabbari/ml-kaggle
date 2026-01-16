# Session Handoff - 2026-01-08

## CRITICAL: Resume from here in next session

### Current Status
- **Best LB**: 0.83862 (mega-blend from Session 5)
- **Our best new submission**: 0.82445 (self-training)
- **Gap to beat**: +0.014 needed

### What We Tried This Session

| Approach | Val Mean AUC | LB | Verdict |
|----------|--------------|-----|---------|
| Self-Training (0.85/0.15 thresholds) | 0.8179 | **0.82445** | Best new approach |
| Self-Training Aggressive (0.70/0.30) | 0.8162 | - | Worse - noise from low thresholds |
| Miracle v2 (17 models) | 0.8147 | - | Not submitted |
| Species-Specific (32 models) | 0.8023 | - | Failed - too few samples per species |
| Various blends | 0.816-0.817 | - | Not better than single best |

### Key Learnings

1. **Self-training works** - semi-supervised learning from test distribution helps
2. **Conservative thresholds better** - 0.85/0.15 > 0.70/0.30 for pseudo-labeling
3. **Species-specific models fail** - not enough samples when split by species
4. **Blending doesn't help much** - similar validation scores, no synergy
5. **Validation is well-calibrated** - Val 0.8179 → LB 0.82445 (+0.006 gap)

### The Original Mega-Blend (0.83862) Used

From `/sata_disk/users/erfan/ml_kaggle/experiments/run_mega_blend.py`:
- **3 models**: PLS-LGB, Species-Global-Blend, Tuned-LGB
- **Rank averaging** for ensemble
- **Species weights**: P.aer 0.05x, K.pn 3.0x, E.coli 1.5x, P.mir 1.5x
- **Tuned LGB params**: n_estimators=350, num_leaves=127, learning_rate=0.02

### What To Try Next (Priority Order)

1. **Run the original mega-blend script** and submit it again
   ```bash
   uv run python experiments/run_mega_blend.py
   ```

2. **Blend mega-blend with self-training** predictions
   - Mega-blend: `experiments/run_mega_blend.py` output
   - Self-training: `/sata_disk/users/erfan/ml_kaggle/outputs/self_training_runs/run_20260108_180837/submissions/submission.csv`

3. **Try TransductiveSVM or Label Propagation** - true transductive learning

4. **Tune self-training thresholds** per antibiotic - weak ones (CIP, LVX, AMC) may need different thresholds

### File Locations

| File | Purpose |
|------|---------|
| `experiments/run_self_training.py` | Self-training pipeline (WORKS) |
| `experiments/run_mega_blend.py` | Original mega-blend (got 0.83862) |
| `experiments/run_miracle_v2.py` | 17-model ensemble |
| `experiments/run_species_specific.py` | 32 species×antibiotic models |
| `experiments/evaluate_blends.py` | Blend evaluation script |

### Best Predictions Available

| Run | Path |
|-----|------|
| Self-Train (0.8179 val) | `outputs/self_training_runs/run_20260108_180837/submissions/submission.csv` |
| Miracle v2 | `outputs/miracle_v2_runs/run_20260108_180757/submissions/best_weighted_all.csv` |
| All blends | `outputs/blend_runs/run_20260108_201645/submissions/` |

### Quick Commands

```bash
# Run original mega-blend
uv run python experiments/run_mega_blend.py

# Submit best self-training
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/self_training_runs/run_20260108_180837/submissions/submission.csv \
  -m 'Self-training'
```
