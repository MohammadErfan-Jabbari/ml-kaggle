# AMR Prediction from MALDI-TOF Mass Spectra

> **Kaggle Competition**: Antimicrobial Resistance Prediction from MALDI-TOF  
> **Task**: Multi-label classification predicting bacterial resistance to 8 antibiotics  
> **Best Leaderboard**: **0.83862** (mega-blend ensemble)  
> **Primary Metric**: Mean AUC across 8 antibiotics

---

## Quick Start

```bash
# Clone and setup
git clone git@github.com:MohammadErfan-Jabbari/ml-kaggle.git
cd ml-kaggle

# Install dependencies (requires uv)
uv sync

# Run the best performing model
uv run python experiments/run_mega_blend.py

# Submit to Kaggle
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/submissions/sub_mega_blend_rank_avg_20260107_2057.csv \
  -m "Mega-blend ensemble"
```

---

## Project Overview

This repository contains work for the **Antimicrobial Resistance (AMR) Prediction from MALDI-TOF** Kaggle competition. The goal is to predict bacterial resistance to 8 antibiotics from MALDI-TOF mass spectrometry data.

### The Challenge

| Aspect | Details |
|--------|---------|
| **Input** | 6,000 MALDI-TOF spectral features (binned mass-tocharge ratios) |
| **Output** | Binary resistance predictions for 8 antibiotics |
| **Species** | 4 bacterial species with significant distribution shift |
| **Train/Test** | 3,360 train samples, 1,000 test samples |
| **Labels** | Partially labeled (42.8% missing for Amoxicillin/Clavulanic acid) |

### 8 Target Antibiotics

| Antibiotic | Class | Resistance Rate |
|------------|-------|-----------------|
| Ampicillin | Aminopenicillin | 88.2% |
| Amoxicillin_Clavulanic_acid | Penicillin + inhibitor | 29.9% |
| Cefotaxime | 3rd gen cephalosporin | 73.8% |
| Cefuroxime | 2nd gen cephalosporin | 76.4% |
| Ciprofloxacin | Fluoroquinolone | 61.1% |
| Ertapenem | Carbapenem | 64.5% |
| Imipenem | Carbapenem | 38.9% |
| Levofloxacin | Fluoroquinolone | 62.4% |

---

## Key Insights (What We Learned)

### 1. Species Distribution Shift is Critical

The training set has a dramatically different species distribution than the test set:

| Species | Train % | Test % | Action |
|---------|---------|--------|--------|
| P. aeruginosa | 43.1% | 3.0% | **Downweight 0.1x** |
| K. pneumoniae | 27.9% | 50.8% | **Upweight 2.0x** |
| E. coli | 16.6% | 26.9% | Upweight 1.5x |
| P. mirabilis | 12.4% | 19.3% | Upweight 1.5x |

**Impact**: Models overfit to P. aeruginosa patterns fail on K. pneumoniae (51% of test).

### 2. Intrinsic Resistance = Free Predictions

Certain species have known intrinsic resistance to specific antibiotics:

| Species | Always Resistant To |
|---------|-------------------|
| P. aeruginosa | Ampicillin, Amox/Clav, Ertapenem, Cefotaxime, Cefuroxime |
| P. mirabilis | Imipenem |

**Result**: 18.8% of test predictions are deterministic based on species alone.

### 3. Validation Strategy Matters

**OOF predictions are unreliable** - they showed ~10% overestimation compared to validation scores.

**Correct approach**: Use validation split that matches test species distribution:
```python
from src.data.dataset import load_validation_split
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
# Train: 2688, Val: 672 (matches test distribution)
```

### 4. Model Selection Insights

| What Works | What Doesn't Work |
|------------|-------------------|
| Rank averaging | Stacking (12.8% overfit) |
| Model diversity (LGB + XGB + CatBoost + MLP) | K.pn AUC optimization alone |
| LightGBM for sparse spectral data | Neural networks alone |
| PLS dimensionality reduction | Unsupervised DR (PCA, KPCA) |

### 5. Label Correlations

Strong correlations indicate shared resistance mechanisms:
- **Levofloxacin <-> Ciprofloxacin**: r=0.925 (same class)
- **Imipenem <-> Ertapenem**: r=0.772 (both carbapenems)
- **Ertapenem <-> Cefotaxime**: r=0.813 (related mechanisms)

---

## Directory Structure

```
ml-kaggle/
├── data/
│   └── processed/          # Validation split cache
├── experiments/            # 30+ experiment scripts
│   ├── run_mega_blend.py           # Best performer (LB: 0.83862)
│   ├── run_self_training.py        # Self-training (LB: 0.82445)
│   ├── run_miracle_v2.py           # 17-model ensemble
│   ├── run_phase[0-5].py           # Sequential experimentation
│   └── transductive_*.py            # Transductive learning attempts
├── knowledge/              # All documentation and insights
│   ├── sessions/          # Detailed session logs
│   ├── insights/          # EDA findings by phase
│   ├── research/          # Biology and MALDI-TOF background
│   ├── hypotheses/        # Hypothesis tracking
│   └── submissions/       # Submission log
├── outputs/
│   ├── eda/               # 40+ visualization plots
│   ├── submissions/       # Kaggle submission CSVs
│   ├── experiments/       # Experiment results JSON
│   ├── self_training_runs/
│   ├── miracle_v2_runs/
│   └── transductive_dr_runs/
├── raw/                   # Competition data (gitignored)
├── scripts/
│   └── eda/               # EDA generation scripts
└── src/
    ├── data/              # Data loading and preprocessing
    ├── features/          # Feature engineering
    ├── models/            # Model architectures
    ├── training/          # Training loops
    ├── inference/         # Prediction utilities
    └── utils/             # Metrics and helpers
```

---

## Competition Results Summary

| Submission | Approach | Val Mean AUC | Leaderboard |
|------------|----------|--------------|-------------|
| #1 | Baseline LightGBM | 0.8030 | 0.8324 |
| #2 | + Reweight + Intrinsic | 0.8030 | 0.8328 |
| #3 | **Mega-blend (rank-avg)** | ~0.90 OOF | **0.83862** |
| #4 | Stacking | 0.95 OOF | 0.8269 (overfit) |
| #5 | Self-training (0.85/0.15) | 0.8179 | 0.82445 |

### Best Model: Mega-Blend

The winning submission combined 3 diverse approaches:

1. **PLS-LGB**: PLS dimensionality reduction + LightGBM
2. **Species-Global-Blend**: Species-aware weighted ensemble  
3. **Tuned-LGB**: Optimized LightGBM parameters

**Ensemble method**: Rank averaging (more robust than weighted averaging)

**Species weights**:
```python
weights = {
    "P.aeruginosa": 0.05,    # Downweight (43% -> 3% in test)
    "K.pneumoniae": 3.0,     # Upweight (28% -> 51% in test)
    "E.coli": 1.5,           # Upweight (17% -> 27% in test)
    "P.mirabilis": 1.5       # Upweight (12% -> 19% in test)
}
```

---

## Essential Code Patterns

### Loading Data with Correct Validation Split

```python
from src.data.dataset import load_validation_split

# Load train/test split matching test species distribution
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()

# Apply species weights
SPECIES_WEIGHTS = {0: 1.5, 1: 2.0, 2: 1.5, 3: 0.1}
weights = np.array([SPECIES_WEIGHTS[s] for s in species_train])
```

### Applying Intrinsic Resistance Rules

```python
def apply_intrinsic_rules(predictions, species_ids):
    """Apply biological rules for known resistance patterns."""
    predictions = predictions.copy()
    
    # P. aeruginosa intrinsic resistance
    pa_mask = (species_ids == 3)
    for ab in ["Ampicillin", "Amoxicillin_Clavulanic_acid", 
               "Ertapenem", "Cefotaxime", "Cefuroxime"]:
        predictions[pa_mask, ANTIBIOTIC_INDICES[ab]] = 1.0
    
    # P. mirabilis intrinsic resistance
    pm_mask = (species_ids == 2)
    predictions[pm_mask, ANTIBIOTIC_INDICES["Imipenem"]] = 1.0
    
    return predictions
```

### Computing Mean AUC (with Missing Labels)

```python
from src.utils.metrics import mean_auc

# Handle NaN labels (semi-supervised)
metrics = mean_auc(y_val, predictions, ANTIBIOTICS)
print(f"Mean AUC: {metrics['mean_auc']:.4f}")
```

---

## Running Experiments

### Reproduce Best Score

```bash
uv run python experiments/run_mega_blend.py
```

### Run Self-Training

```bash
uv run python experiments/run_self_training.py
```

### Generate EDA Plots

```bash
# Phase 1: Data profiling
uv run python scripts/eda/phase1_data_profiling.py

# Phase 2: Feature analysis
uv run python scripts/eda/phase2_feature_analysis.py

# Phase 3: Target analysis
uv run python scripts/eda/phase3_target_analysis.py

# ... etc for phases 4-8
```

---

## Poster Assets

The repository includes comprehensive visualizations suitable for academic posters:

### EDA Plots (outputs/eda/)

| Category | Files |
|----------|-------|
| **Species Analysis** | species_distribution.png, mean_spectrum_by_species.png |
| **Resistance Patterns** | resistance_by_species.png, missing_label_heatmap.png |
| **Feature Analysis** | feature_correlation_sampled.png, variance_by_feature.png |
| **Correlations** | target_correlation.png, antibiotic_dendrogram.png |
| **Class Balance** | target_class_balance.png, resistance_count_distribution.png |
| **Biomarkers** | biomarker_features_by_antibiotic.png, intrinsic_resistance_patterns.png |

### Key Statistics

- **Total samples**: 3,360 train, 1,000 test
- **Features**: 6,000 MALDI-TOF spectral bins
- **Species**: 4 bacterial species with distribution shift
- **Best AUC**: 0.83862 (mega-blend ensemble)
- **Feature sparsity**: 93.3% zeros (natural for MALDI-TOF)

---

## Dependencies

```toml
[dependencies]
catboost = ">=1.2.8"
kaggle = ">=1.8.3"
lightgbm = ">=4.6.0"
matplotlib = ">=3.10.8"
mlflow = ">=3.8.1"
numpy = ">=2.4.0"
pandas = ">=2.3.3"
pyyaml = ">=6.0.3"
rich = ">=14.2.0"
scikit-learn = ">=1.8.0"
scipy = ">=1.16.3"
seaborn = ">=0.13.2"
torch = ">=2.9.1"
xgboost = ">=3.1.2"
```

Install with:
```bash
uv sync
```

---

## Competition Context

### MALDI-TOF Mass Spectrometry

MALDI-TOF (Matrix-Assisted Laser Desorption/Ionization Time-of-Flight) mass spectrometry is a rapid microbial identification technique that measures the mass-to-charge ratio of ionized proteins.

**Key characteristics**:
- Mass range: 2,000-20,000 Da
- Primary signals: Ribosomal proteins
- Our data: 6,000 binned features (~3 Da per bin)

### Why AMR Prediction Matters

Antimicrobial resistance is a global health threat. Rapid prediction of resistance patterns from routine MALDI-TOF data could:
1. Reduce turnaround time for AST results
2. Enable earlier appropriate antibiotic therapy
3. Improve patient outcomes
4. Reduce healthcare costs

---

## References

### Competition
- [Kaggle Competition Page](https://www.kaggle.com/competitions/antimicrobial-resistance-prediction-from-maldi-tof)

### Key Papers
- DRIAMS-2022: https://www.nature.com/articles/s41591-021-01619-9
- maldi-learn: https://github.com/BorgwardtLab/maldi_amr
- MSDeepAMR (2024): 1D CNN approach, AUROC 0.82-0.93

---

## License

This project is for educational and research purposes.

---

## Contact

**Mohammad Erfan Jabbari**  
GitHub: [@MohammadErfan-Jabbari](https://github.com/MohammadErfan-Jabbari)

---

*Last updated: January 2026*
