# Course Methods - Quick Reference

## Relevant Methods for This Competition

### PLS (Partial Least Squares)
- Use `n_components = n_classes - 1 = 7` (or up to 20)
- Supervised dimensionality reduction
- `from sklearn.cross_decomposition import PLSRegression`

### Ensembles
- **Bagging**: Variance reduction for small data
- **GBM/LightGBM**: Best for sparse data (93% zeros)
- **NO stacking** - proven to overfit here

### Feature Selection
- Variance threshold (remove var < 1e-5)
- LGB feature importance

### Semi-Supervised (for missing labels)
- K-Means clustering to propagate labels
- PLS regression to predict missing

---

## Code Patterns

```python
# PLS + LightGBM Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pls = PLSRegression(n_components=20)
X_pls = pls.fit_transform(X_scaled, y)
model = lgb.LGBMClassifier(n_estimators=200)
model.fit(X_pls, y, sample_weight=weights)
```

---

## Don't Use
- Kernel PCA with RBF (degrades on sparse data)
- Deep neural networks (overfit on 3360 samples)
- Complex stacking (12.8% overfit gap)
