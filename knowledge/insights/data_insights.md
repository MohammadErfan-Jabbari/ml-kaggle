# Data Insights - AMR Prediction Competition

## Dataset Overview

| Metric | Train | Test |
|--------|-------|------|
| Samples | 3,360 | 1,000 |
| MALDI Features | 6,000 | 6,000 |
| Targets | 8 antibiotics | To predict |
| Species | 4 | 4 |

## Critical Finding: Species Distribution Shift

**Train vs Test species distribution:**
| Species | Train % | Test % | Shift |
|---------|---------|--------|-------|
| E. coli (0) | 16.6% | 26.9% | +10.3% |
| K. pneumoniae (1) | 27.9% | 50.8% | +22.9% |
| P. mirabilis (2) | 12.4% | 19.3% | +6.9% |
| **P. aeruginosa (3)** | **43.1%** | **3.0%** | **-40.1%** |

**Implication**: P. aeruginosa dominates training but is rare in test. Models must generalize to other species!

## Target Distribution

| Antibiotic | Total Labeled | Resistant % | Missing |
|------------|---------------|-------------|---------|
| Ampicillin | 3332 | 88.2% | 28 |
| Levofloxacin | 3271 | 52.2% | 89 |
| Ciprofloxacin | 3274 | 53.4% | 86 |
| Imipenem | 3249 | 55.5% | 111 |
| Amox/Clav | 1921 | 29.9% | **1439** |
| Ertapenem | 3360 | 46.3% | 0 |
| Cefotaxime | 3357 | 56.2% | 3 |
| Cefuroxime | 3327 | 75.2% | 33 |

**Key observation**: Amoxicillin_Clavulanic_acid has 43% missing labels!

## Resistance Rates by Species

### E. coli (species_id=0)
- Ampicillin: 67.0% R
- Imipenem: **0.4% R** (very susceptible)
- Ertapenem: 1.6% R
- Cefuroxime: 84.2% R

### K. pneumoniae (species_id=1)
- Ampicillin: **99.9% R** (near-total resistance)
- Imipenem: 5.5% R
- Ertapenem: 10.6% R

### P. mirabilis (species_id=2)
- Imipenem: **97.2% R** (unusual, intrinsic)
- Ertapenem: 0.2% R (susceptible)
- Ampicillin: 47.5% R

### P. aeruginosa (species_id=3) - CRITICAL
- Ampicillin: **100% R**
- Ertapenem: **100% R**
- Cefotaxime: **100% R**
- Cefuroxime: **100% R**
- Amox/Clav: **100% R** (only 54 labeled)
- Imipenem: 99.9% R

**→ For P. aeruginosa, 5/8 antibiotics can be predicted as resistant deterministically!**

## Label Correlations

Strong positive correlations (shared mechanisms):
- Levofloxacin ↔ Ciprofloxacin: **0.92** (fluoroquinolones)
- Ertapenem ↔ Cefotaxime: **0.81**
- Imipenem ↔ Ertapenem: **0.77** (carbapenems)
- Cefotaxime ↔ Cefuroxime: **0.66** (cephalosporins)

## Feature Characteristics

| Statistic | Value |
|-----------|-------|
| Features | 6,000 binned intensities |
| Min value | 0.0 |
| Max value | 3.0115 |
| Mean | 0.0869 |
| Sparsity | **93.4% zeros** |
| Constant features | 365 |

**→ Data is very sparse with many zero values. Consider handling zeros specially.**

## Missing Label Patterns

| # Missing Labels | Count | % |
|------------------|-------|---|
| 0 (all labeled) | 1,800 | 53.6% |
| 1 | 1,396 | 41.5% |
| 2 | 109 | 3.2% |
| 3+ | 55 | 1.6% |

Most samples have 0 or 1 missing label. The missing labels are concentrated in Amoxicillin_Clavulanic_acid.

## Preprocessing Recommendations

1. **Normalization**: StandardScaler or log1p transform (handle sparsity)
2. **Feature filtering**: Remove 365 constant features
3. **Species handling**:
   - Use embedding (already in baseline)
   - Consider species-stratified validation
   - P. aeruginosa needs special handling (intrinsic resistance)
4. **Missing labels**: Mask in loss function, consider pseudo-labeling

## Modeling Implications

1. **Species is the most predictive feature** for many antibiotics
2. **P. aeruginosa dominating train is misleading** - test has different distribution
3. **Fluoroquinolones** (Levo/Cipro) should be modeled together due to r=0.92
4. **Amox/Clav** has most missing data - may need semi-supervised approach
5. **Simple species-based rules** can achieve high accuracy for some targets

## Quick Wins

1. For P. aeruginosa: predict 1 for Ampicillin, Ertapenem, Cefotaxime, Cefuroxime
2. Stratify validation by species to avoid overfitting to P. aeruginosa
3. Weight non-P. aeruginosa samples higher during training
