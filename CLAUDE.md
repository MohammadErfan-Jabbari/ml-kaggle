# AMR Prediction from MALDI-TOF

## Quick Start

```bash
# Setup
uv sync

# Train model
uv run python src/training/train.py

# Generate submission
uv run python src/inference/predict.py --checkpoint outputs/models/best.pt

# Submit to Kaggle
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/submissions/submission.csv -m "Description"
```

---

## Session Protocol

**At session start**, read these files in order:
1. `knowledge/SESSION_STATE.md` - Current state and next actions (contains link to strategy)
2. `knowledge/EDA_CONCLUSIONS_STRATEGY.md` - ⭐ MANDATORY: All modeling decisions must follow this
3. `knowledge/COURSE_METHODS_REFERENCE.md` - Methods/techniques from course (semi-supervised, feature engineering, etc.)
4. `knowledge/hypotheses/hypothesis_tracker.md` - Active hypotheses

**NOTE**: The EDA Conclusions document contains 5 Critical Truths that override any default approaches. Read it before coding anything.

**During session**:
- Update `hypothesis_tracker.md` as experiments complete
- Log significant findings in `knowledge/insights/`
- Track experiments in `knowledge/experiments/experiment_log.md`

**At session end**:
- Update `SESSION_STATE.md` with current state
- Create session log in `knowledge/sessions/YYYY-MM-DD.md`
- Update decision tree if new branch explored

---

## Project Overview

| Aspect | Value |
|--------|-------|
| **Task** | Multi-label classification (8 antibiotics) |
| **Data** | 3360 train / 1000 test samples, 6000 MALDI features |
| **Metric** | Mean AUC across 8 antibiotics |
| **Challenge** | Semi-supervised (missing labels), species distribution shift |

**Critical Insight**: P. aeruginosa is 43% of train but only 3% of test. Don't overfit to it.

---

## Tech Stack

| Tool | Purpose | Command |
|------|---------|---------|
| uv | Package manager | `uv sync`, `uv add <pkg>` |
| PyTorch | Deep learning | - |
| scikit-learn | Traditional ML | - |
| LightGBM | Gradient boosting | `uv add lightgbm` |
| MLflow | Experiment tracking | `uv run mlflow ui` |
| Kaggle CLI | Submissions | `kaggle competitions submit ...` |

---

## Directory Structure

```
ml_kaggle/
├── CLAUDE.md              # This file - how to work here
├── knowledge/             # Accumulated knowledge (READ FIRST)
│   ├── SESSION_STATE.md   # Current state - START HERE
│   ├── APPROACH_PLAN.md   # Phased competition strategy
│   ├── COURSE_METHODS_REFERENCE.md  # Methods from course notebooks
│   ├── Ml-course_notebooks/  # Professor's Jupyter notebooks
│   ├── research/          # Domain knowledge
│   ├── insights/          # Data analysis findings
│   ├── experiments/       # Experiment logs and results
│   ├── hypotheses/        # Hypothesis tracking
│   └── sessions/          # Session logs
├── raw/                   # Competition data
├── src/                   # Source code
│   ├── data/dataset.py    # Data loading, MaldiDataset
│   ├── models/            # Model architectures
│   ├── training/train.py  # Training loop
│   ├── inference/predict.py # Submission generation
│   └── utils/metrics.py   # Evaluation metrics
├── configs/               # Hyperparameter configs
├── scripts/               # Utility scripts (EDA, etc.)
└── outputs/               # Models, submissions (gitignored)
```

---

## Data Quick Reference

| File | Samples | Description |
|------|---------|-------------|
| `raw/train.csv` | 3360 | Features + labels (some NaN) |
| `raw/test.csv` | 1000 | Features only |

**Targets**: `Ampicillin`, `Levofloxacin`, `Ciprofloxacin`, `Imipenem`, `Amoxicillin_Clavulanic_acid`, `Ertapenem`, `Cefotaxime`, `Cefuroxime`

**Species**: 0=E.coli, 1=K.pneumoniae, 2=P.mirabilis, 3=P.aeruginosa

---

## Code Modules

| Module | Key Functions |
|--------|---------------|
| `src/data/dataset.py` | `load_train_data()`, `MaldiDataset`, `get_dataloaders()` |
| `src/models/baseline.py` | `MLPBaseline`, `create_model(config)` |
| `src/training/train.py` | Training loop with early stopping |
| `src/inference/predict.py` | Generate Kaggle submissions |
| `src/utils/metrics.py` | `mean_auc()` - handles NaN labels |

---

## Verification Commands

```bash
# Verify training works
uv run python src/training/train.py

# Verify submission format
uv run python -c "
import pandas as pd
sub = pd.read_csv('outputs/submissions/submission.csv')
sample = pd.read_csv('raw/sample_submission.csv')
print('Shape match:', sub.shape == sample.shape)
print('Columns match:', list(sub.columns) == list(sample.columns))
"

# Quick data check
uv run python -c "
import pandas as pd
train = pd.read_csv('raw/train.csv')
print(f'Train: {train.shape}')
print(f'Missing labels: {train.iloc[:, -8:].isna().sum().to_dict()}')
"
```

---

## Knowledge System Guide

The `knowledge/` folder preserves context across sessions. Use it as follows:

### When Starting a New Task
1. Check `SESSION_STATE.md` for current priorities
2. Read `EDA_CONCLUSIONS_STRATEGY.md` for mandatory constraints (species shift, etc.)
3. Check `COURSE_METHODS_REFERENCE.md` for relevant course methods (semi-supervised, feature engineering, etc.)
4. Review `hypothesis_tracker.md` for what's been tried
5. Consult `research/` for domain knowledge if needed

### When Exploring Options
1. Check `experiments/decision_tree.md` for paths already explored
2. Review `experiments/experiment_log.md` for past results
3. Avoid re-running failed approaches without new insights

### When Implementing
1. Update hypothesis status to `in_progress`
2. Log experiment config and results
3. Update `insights/` with new findings

### When Finishing Session
1. Update `SESSION_STATE.md` with what was accomplished
2. Create session log if significant work done
3. Update hypothesis tracker with outcomes
4. Note any new hypotheses discovered

---

## Key Constraints

1. **NaN Labels**: Use masked loss - never compute loss on NaN targets
2. **Species Shift**: Validate on species-stratified folds
3. **Small Data**: 3360 samples - regularization critical, GBM may beat NN
4. **Sparse Features**: 93% zeros in MALDI features

---

## Links to Knowledge

| Topic | Location |
|-------|----------|
| Current state & next steps | `knowledge/SESSION_STATE.md` |
| Competition strategy | `knowledge/APPROACH_PLAN.md` |
| **Course methods reference** | `knowledge/COURSE_METHODS_REFERENCE.md` ⭐ |
| **EDA Conclusions (5 Truths)** | `knowledge/EDA_CONCLUSIONS_STRATEGY.md` ⭐ |
| MALDI-TOF fundamentals | `knowledge/research/maldi_tof.md` |
| Species-specific resistance | `knowledge/research/amr_biology.md` |
| ML approaches | `knowledge/research/ml_sota.md` |
| Data insights | `knowledge/insights/data_insights.md` |
| Hypothesis queue | `knowledge/hypotheses/hypothesis_tracker.md` |
| Experiment history | `knowledge/experiments/experiment_log.md` |
| Decision tree | `knowledge/experiments/decision_tree.md` |

**⭐ = Read before modeling**
