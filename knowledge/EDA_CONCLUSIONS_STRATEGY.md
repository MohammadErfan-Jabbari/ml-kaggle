# EDA Conclusions - Essential Facts Only

## The 5 Critical Truths

### 1. Species Distribution Shift (CRITICAL)
| Species | Train | Test | Weight |
|---------|-------|------|--------|
| P. aeruginosa | 43.1% | 3.0% | **0.1x** |
| K. pneumoniae | 27.9% | 50.8% | **2.0x** |
| E. coli | 16.6% | 26.9% | 1.5x |
| P. mirabilis | 12.4% | 19.3% | 1.5x |

### 2. Intrinsic Resistance = Free Predictions (18.8% of test)
| Species | Antibiotics | Predict |
|---------|------------|---------|
| P. aeruginosa | Ampicillin, Amox/Clav, Ertapenem, Cefotaxime, Cefuroxime | **1.0** |
| P. mirabilis | Imipenem | **1.0** |

### 3. Extreme Feature Sparsity
- 93.3% zeros in MALDI features
- ~765 constant features (remove with var < 1e-5)
- **LightGBM >> Neural Networks** for this data

### 4. Missing Labels
- Amox/Clav: 42.8% missing (96.3% for P. aeruginosa)
- Use masked loss, never compute loss on NaN

### 5. Antibiotic Correlations
- Levofloxacin ↔ Ciprofloxacin: 0.925 (same class)
- Multi-task learning justified

---

## Metric Alignment (CRITICAL)

**Primary metric**: Val Mean AUC = average of AUC across all 8 antibiotics

**DO NOT** optimize K.pn AUC alone - proven to hurt LB

---

## Mandatory Code Patterns

### Sample Weighting
```python
SPECIES_WEIGHTS = {0: 1.5, 1: 2.0, 2: 1.5, 3: 0.1}
weights = np.array([SPECIES_WEIGHTS[s] for s in species_train])
```

### Validation Split
```python
from src.data.dataset import load_validation_split
X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
```

### Intrinsic Rules
```python
def apply_intrinsic_rules(predictions, species_ids):
    predictions = predictions.copy()
    pa_mask = (species_ids == 3)
    for ab in ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"]:
        predictions[pa_mask, ANTIBIOTIC_INDICES[ab]] = 1.0
    pm_mask = (species_ids == 2)
    predictions[pm_mask, ANTIBIOTIC_INDICES["Imipenem"]] = 1.0
    return predictions
```

---

## Model Choice: LightGBM

| Factor | LightGBM | Neural Net |
|--------|----------|------------|
| 93% sparsity | ✓ Native | ✗ |
| 3360 samples | ✓ | ✗ Overfit |
| Missing labels | ✓ Native | ✗ |
| Speed | ✓ Fast | ✗ Slow |

---

## What Works / What Doesn't

| Works | Doesn't Work |
|-------|--------------|
| Rank averaging | Stacking (overfits) |
| Model diversity | K.pn optimization |
| Species weighting | Complex feature eng |
| Intrinsic rules | Single models |
