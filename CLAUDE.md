# AMR Prediction from MALDI-TOF - Competition Guide

## ⚠️ START HERE - Read `knowledge/sessions/2026-01-08-session-handoff.md`

## Quick Commands
```bash
uv sync                                    # Setup
uv run python experiments/run_mega_blend.py   # Original best (LB=0.83862)
uv run python experiments/run_self_training.py  # Self-training (LB=0.82445)
```

## Competition Status
| Metric | Value |
|--------|-------|
| **Best LB** | 0.83862 (mega-blend from Session 5) |
| **Our New Best** | 0.82445 (self-training) |
| **Gap to Beat** | +0.014 needed |
| **Primary Metric** | **Val Mean AUC** (avg across 8 antibiotics) |

## Critical Rules (NON-NEGOTIABLE)

### 1. Use Val Mean AUC, NOT K.pn AUC
- Leaderboard = mean AUC across 8 antibiotics
- Optimizing K.pn alone HURTS LB (proven: +5.6% K.pn → -1.6% LB)

### 2. Use Validation Split (matches test distribution)
```python
from src.data.dataset import load_validation_split
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
# Train: 2688, Val: 672 (matches test species distribution)
```

### 3. Species Shift is Critical
| Species | Train | Test | Action |
|---------|-------|------|--------|
| P. aeruginosa | 43% | 3% | Downweight 0.1x |
| K. pneumoniae | 28% | 51% | Upweight 2.0x |
| E. coli | 17% | 27% | Upweight 1.5x |
| P. mirabilis | 12% | 19% | Upweight 1.5x |

### 4. Intrinsic Resistance Rules (free predictions)
- P. aeruginosa → 1.0 for: Ampicillin, Amox/Clav, Ertapenem, Cefotaxime, Cefuroxime
- P. mirabilis → 1.0 for: Imipenem

### 5. NO Stacking (overfits massively)
- OOF=0.9545 → LB=0.8269 (12.8% gap!)
- Use simple averaging or rank averaging only

## Miracle Blend Script
Location: `experiments/run_miracle_v2.py`

**What it does:**
1. Loads data with proper validation split
2. Builds 17 diverse models (LightGBM, XGBoost, CatBoost, MLP, PLS+LGB)
3. Creates 4 ensemble variants (rank-avg, weighted-avg, top-N, meta-blend)
4. Evaluates with **Val Mean AUC** (the LB metric)
5. Saves submissions to `outputs/submissions/`

**Run time:** ~30-40 minutes

## Data
| File | Samples | Notes |
|------|---------|-------|
| train.csv | 3360 | 6000 MALDI features, 8 antibiotic labels (some NaN) |
| test.csv | 1000 | Species shift: 51% K.pn (vs 28% train) |

## Key Files
| File | Purpose |
|------|---------|
| `experiments/run_miracle_v2.py` | **MAIN: Ultimate ensemble** |
| `experiments/run_mega_blend.py` | Previous best (LB=0.83862) |
| `src/data/dataset.py` | Data loading + validation split |
| `knowledge/SESSION_STATE.md` | Current state (condensed) |

## Submission
```bash
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/submissions/sub_miracle_v2_xxx.csv \
  -m "Miracle v2 ensemble"
```

## What's Been Tried
| Approach | Val Mean AUC | LB | Verdict |
|----------|--------------|-----|---------|
| Baseline LightGBM | 0.80 | 0.8324 | OK |
| + Intrinsic rules | 0.80 | 0.8328 | Minimal |
| **Mega-blend rank-avg** | ~0.90 OOF | **0.83862** | **BEST LB** |
| Stacking | 0.95 OOF | 0.8269 | OVERFIT |
| Self-Training (0.85/0.15) | 0.8179 | 0.82445 | Best new approach |
| Self-Training Aggressive | 0.8162 | - | Worse |
| Miracle v2 (17 models) | 0.8147 | - | Not submitted |
| Species-Specific (32) | 0.8023 | - | Failed |
| **Transductive PCA 100** | 0.7758 | - | **FAILED - PCA hurts** |
| **Transductive PCA 200** | 0.7901 | - | **FAILED - PCA hurts** |
| **PCA 200 + MLP only** | 0.7754 | - | **FAILED - PCA hurts** |

## Next Steps to Try
1. **Re-run mega-blend** - the original that got 0.83862
2. **Blend mega-blend + self-training** predictions
3. **Per-antibiotic threshold tuning** for self-training

## Don't Waste Time On
- Stacking/meta-learners (overfit)
- Optimizing K.pn AUC alone (hurts mean)
- Complex feature engineering (diminishing returns)
- Neural networks alone (need ensemble)
- **Unsupervised DR (PCA, KPCA, PPCA) - PROVEN TO HURT PERFORMANCE**
