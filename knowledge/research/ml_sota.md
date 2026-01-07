# State-of-the-Art ML Approaches for MALDI-TOF AMR Prediction

## Model Architecture Comparison

### Traditional ML (Often Wins on Tabular)
| Algorithm | Strengths | Best For | Expected AUC |
|-----------|-----------|----------|--------------|
| LightGBM | Fast, handles imbalance, feature importance | E. coli, general | 0.74-0.80 |
| XGBoost | Robust, regularized | General use | 0.74-0.80 |
| Random Forest | Interpretable, robust | Baseline | 0.70-0.78 |
| Logistic Regression | Calibrated probabilities | Feature analysis | 0.65-0.75 |

### Deep Learning
| Architecture | Strengths | Best For | Expected AUC |
|--------------|-----------|----------|--------------|
| MLP | Non-linear patterns | With good features | 0.74-0.80 |
| 1D CNN | Local spectral patterns | Raw/binned spectra | 0.82-0.93 |
| Transformer | Peak relationships | Large datasets | Improving |
| Autoencoder | Representation learning | Pre-training | N/A (feature extraction) |

## When to Use What (Research Summary)

**Gradient Boosting wins when:**
- Dataset < 5000 samples ← **Our case**
- Features have skewed distributions
- Need interpretability
- Quick iteration needed

**Deep Learning wins when:**
- Dataset > 10,000 samples
- Raw spectral data available
- Transfer learning possible
- Complex feature interactions

## Recommended Architectures for Our Competition

### 1. LightGBM Baseline (Start Here)
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'class_weight': 'balanced'
}
```

### 2. Multi-Scale 1D CNN
- Kernel sizes: [5, 11, 21] to capture different peak widths
- Species embedding concatenated after conv layers
- Dropout 0.3-0.5, BatchNorm after each conv
- Expected improvement: +5-10% over MLP

### 3. Attention MLP
- Feature attention layer to identify important m/z regions
- Interpretable: attention weights show which peaks matter
- Good for publication/understanding

### 4. Ensemble Strategy
```
Final = 0.4 * LightGBM + 0.3 * XGBoost + 0.3 * CNN
```

## Handling Missing Labels

### Masked BCE Loss (CRITICAL)
```python
def masked_bce_loss(logits, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask]
    )
```

### Pseudo-Labeling for Semi-Supervised
1. Train on labeled data
2. Predict unlabeled with threshold > 0.9
3. Add high-confidence pseudo-labels to training
4. Repeat with decreasing threshold

### Label Correlation Exploitation
- **Classifier Chains**: Order by label frequency, condition on previous predictions
- **Multi-Task Learning**: Shared backbone, separate heads per antibiotic
- **Group antibiotics**: Fluoroquinolones together, carbapenems together

## Cross-Validation Strategy

### Multi-Label Stratified K-Fold
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Species-Aware Stratification
- Ensure each fold has representative samples from all 4 species
- Critical because species distribution differs between train/test

## Data Augmentation for Spectra

| Technique | Implementation | Effect |
|-----------|----------------|--------|
| Gaussian Noise | `x + noise * 0.02` | Robustness to measurement noise |
| Intensity Scaling | `x * uniform(0.9, 1.1)` | Handle intensity variation |
| Peak Shifting | `roll(x, shift=±3)` | Handle m/z calibration drift |
| Mixup | `λ*x1 + (1-λ)*x2` | Regularization, smooth boundaries |
| Dropout | Zero random bins | Robustness to missing peaks |

## Feature Engineering

### Peak-Based Features
- Peak count, mean height, max height
- Peak width, area under curve
- Spectral entropy

### Regional Features
- Low mass (2000-5000 Da): Small proteins
- Mid mass (5000-10000 Da): Main region
- High mass (10000-20000 Da): Large proteins
- Ratios between regions

### Dimensionality Reduction
- Remove near-constant features (std < 0.01)
- PCA (keep 95% variance)
- Select top-k by mutual information per target

## Key GitHub Repositories

| Repository | Use For |
|------------|---------|
| [BorgwardtLab/maldi_amr](https://github.com/BorgwardtLab/maldi_amr) | Production pipeline, preprocessing |
| [BorgwardtLab/maldi-learn](https://github.com/BorgwardtLab/maldi-learn) | Sklearn-compatible tools |
| [gdewael/maldi-nn](https://github.com/gdewael/maldi-nn) | Pre-trained transformers |
| [pytorch-tabular](https://github.com/pytorch-tabular/pytorch_tabular) | Tabular deep learning |
| [iterative-stratification](https://github.com/trent-b/iterative-stratification) | Multi-label CV |
