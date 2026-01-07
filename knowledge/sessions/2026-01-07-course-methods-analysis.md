# Session Log: 2026-01-07

## Overview
**Type**: Course Methods Analysis
**Agent**: claude-opus-4.5
**Duration**: Session 3

---

## Objectives
1. Extract methods and techniques from professor's ML course notebooks
2. Map course methods to AMR prediction competition challenges
3. Create reference document for future sessions
4. Update project instructions to include course methods

---

## Actions Completed

### 1. Course Notebook Analysis
**Approach**: Used haiku agents (gemini-3-flash, 1M context) to analyze 13 notebooks in parallel

| Notebook | Key Methods | AMR Application |
|----------|-------------|-----------------|
| NoveltyDetection_1SVM | One-Class SVM, PCA+SVM pipeline | Species shift detection |
| KernelMethods | SVM, KRR, Kernel Logistic Regression | Sparse non-linear features |
| GP_regression | Gaussian Processes, LML, Uncertainty | Small data robustness |
| Clustering_Kmeans | K-Means, Silhouette analysis | Semi-supervised labeling |
| FE_homework | PLS, CCA, Nyström, RBF Sampler | Supervised feature extraction |
| Intro_SVMs | LinearSVC, One-vs-Rest | Multi-label baseline |
| GP_withGPy | Kernel engineering, ARD, Sparse GP | Feature selection for 6000 peaks |
| Ensembles | Bagging, AdaBoost, XGBoost | LightGBM baseline |
| Feature_extraction | PLS, CCA, Kernel PLS/CCA | Resistance-specific peaks |
| MNIST_homework | PCA + Classifier pipeline | Template for MALDI pipeline |
| Prob_PCA | PPCA, Generative sampling | Noise reduction, augmentation |
| Bayesian_regression (prof) | Kernel Bayes, Evidence max | Uncertainty-based OOD |
| Bayesian_regression (student) | Sequential updating | Alternative hyperparameter tuning |

### 2. Created Knowledge Base
**File**: `knowledge/COURSE_METHODS_REFERENCE.md`

Contents:
- Quick reference table of methods by category
- Detailed breakdown of each notebook
- Recommended toolkit for AMR competition
- Semi-supervised strategies for missing labels
- Dimensionality reduction strategy
- Parameter tuning priorities
- Cross-reference table linking concepts to notebooks

### 3. Updated Project Instructions
**File**: `CLAUDE.md`

Changes:
- Added COURSE_METHODS_REFERENCE.md to session protocol (#3)
- Updated directory structure to include Ml-course_notebooks/
- Enhanced "When Starting a New Task" workflow
- Added course methods to Links to Knowledge table with ⭐ markers

---

## Key Insights

### Course → Competition Mapping

**For Semi-Supervised Learning (42.8% missing labels)**:
- K-Means clustering → Group samples, propagate labels within clusters
- Kernel similarity → Weighted label propagation from neighbors
- PLS regression → Use labeled samples to predict missing

**For High-Dimensional Features (6000 MALDI peaks)**:
- PCA → Reduce to 200-500, remove noise
- PLS/CCA (supervised) → Find resistance-specific peaks
- ARD (from GP) → Automatic feature selection

**For Species Distribution Shift**:
- One-Class SVM → Detect OOD test samples
- Bayesian uncertainty → Flag low-confidence predictions
- Sample reweighting → Down-weight P. aeruginosa (0.3x)

**For Baseline Modeling**:
- LightGBM/GBM with Bagging → Variance reduction
- max_features sampling → Increase diversity, prevent overfitting

### Critical Emphasis from Course
**Leakage-safe pipelines**: Fit feature extractors (PCA/PLS/CCA) only within CV folds, not on entire dataset. This is critical for small data (3360 samples) to avoid optimistic validation scores.

Template from course:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', GradientBoostingClassifier())
])
```

---

## Deliverables

| File | Purpose |
|------|---------|
| `knowledge/COURSE_METHODS_REFERENCE.md` | Comprehensive methods reference |
| `knowledge/sessions/2026-01-07-course-methods-analysis.md` | This session log |
| `CLAUDE.md` | Updated with course methods integration |

---

## Next Session

**Priority**: Implement baseline infrastructure (from SESSION_STATE.md)

1. Core utilities (`src/utils/`):
   - `MaskedBCEWithLogitsLoss` for NaN labels
   - `get_species_stratified_cv()` for proper splits
   - `get_sample_weights()` for P. aeruginosa downweighting
   - `intrinsic_resistance_rules()` for rule-based predictions

2. LightGBM baseline (`src/models/lightgbm_baseline.py`):
   - Remove constant features
   - Train 8 separate models (one per antibiotic)
   - Species-stratified 5-fold CV
   - Sample weights and per-species metrics

3. Hybrid inference (`src/inference/predict.py`):
   - Apply intrinsic resistance rules
   - LightGBM for remaining predictions
   - Generate submission.csv

4. First Kaggle submission

---

## Status

- ✅ EDA Complete (8 phases, 45+ analyses)
- ✅ Course methods analyzed
- ✅ Knowledge base created
- ⏳ Baseline implementation (next)

**Hypothesis Queue**: H1 (NaN handling) → H2 (LightGBM baseline) → H3 (Hybrid rule + ML)
