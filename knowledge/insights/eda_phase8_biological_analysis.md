# EDA Phase 8: Biological/Domain-Specific Analysis Report

**Generated:** 2026-01-07
**Data:** /sata_disk/users/erfan/ml_kaggle/raw/train.csv
**Samples:** 3360
**Focus:** Antibiotic classes, intrinsic resistance, species-specific mechanisms

---

## Executive Summary

This report analyzes the biological and domain-specific aspects of AMR prediction:

1. **Antibiotic Class Relationships**: Within-class correlations stronger than between-class
2. **Multi-Drug Resistance (MDR)**: 65.5% of samples resistant to ≥3 classes
3. **Intrinsic Resistance Patterns**: Documented for P. aeruginosa and P. mirabilis
4. **Species-Specific Modeling**: 81.2% of species-antibiotic combinations require ML
5. **MALDI-TOF Biomarkers**: Top discriminative features identified

**Key Insight**: Significant modeling complexity can be reduced by using rule-based predictions for intrinsic resistance cases.

---

## 8.1 Antibiotic Class Analysis

### Class Groupings

| Class | Antibiotics |
|-------|-------------|
| Fluoroquinolones | Levofloxacin, Ciprofloxacin |
| Carbapenems | Imipenem, Ertapenem |
| Cephalosporins | Cefotaxime, Cefuroxime |
| Penicillin+Inhibitor | Amoxicillin_Clavulanic_acid |
| Penicillins | Ampicillin |

### Within-Class vs Between-Class Correlations

| Metric | Value |
|--------|-------|
| Mean within-class correlation | 0.785 |
| Mean between-class correlation | 0.401 |
| Difference | 0.383 |

**Interpretation**: Same-class antibiotics show stronger correlations, indicating shared resistance mechanisms. This supports:
- Multi-task learning architectures
- Feature sharing between correlated targets
- Transfer learning within antibiotic classes

### Cross-Resistance Patterns

Cross-resistance is high within classes, particularly:
- **Fluoroquinolones**: P(Ciprofloxacin | Levofloxacin) > 95%
- **Carbapenems**: Shared resistance mechanisms

---

## 8.2 Multi-Drug Resistance (MDR) Analysis

### Definition

MDR = Resistance to ≥3 antibiotic classes

### MDR Prevalence by Species

| Species | Total Samples | MDR Samples | MDR Rate |
|---------|--------------|-------------|----------|
| E. coli | 559 | 248 | 44.4% |
| K. pneumoniae | 939 | 326 | 34.7% |
| P. mirabilis | 415 | 180 | 43.4% |
| P. aeruginosa | 1447 | 1447 | 100.0% | |

### Modeling Implications

1. **MDR samples**: Require careful handling to avoid overfitting
2. **Non-MDR samples**: May be easier to predict
3. **Species-specific MDR patterns**: P. aeruginosa has highest MDR rate (partly due to intrinsic resistance)

---

## 8.3 Intrinsic Resistance Patterns

### Intrinsic Resistance by Species

| Species | Antibiotic | Observed Rate | Expected |
|---------|------------|---------------|----------|
| **P. aeruginosa** | Ampicillin | ~100% | Intrinsic ✓ |
| **P. aeruginosa** | Amoxicillin_Clavulanic_acid | ~100% | Intrinsic ✓ |
| **P. aeruginosa** | Ertapenem | ~100% | Intrinsic ✓ |
| **P. aeruginosa** | Cefotaxime | ~100% | Intrinsic ✓ |
| **P. aeruginosa** | Cefuroxime | ~100% | Intrinsic ✓ |
| **P. mirabilis** | Imipenem | ~100% | Intrinsic ✓ |
| **E. coli** | All | Variable | No intrinsic resistance ✓ |
| **K. pneumoniae** | All | Variable | No intrinsic resistance ✓ |

### Biological Mechanisms

**P. aeruginosa intrinsic resistance:**
- Chromosomal AmpC beta-lactamase → resistance to penicillins and cephalosporins
- Poor porin entry + efflux pumps → resistance to carbapenems (except imipenem)
- Natural resistance to beta-lactamase inhibitors

**P. mirabilis intrinsic resistance:**
- Natural resistance to imipenem

### Modeling Strategy

For intrinsically resistant combinations, **use deterministic rules (predict 1.0)** rather than ML models.

---

## 8.4 Species-Specific Resistance Variability

### Trivial vs Non-Trivial Cases

| Species | Antibiotics Requiring ML | Antibiotics (Rule-Based) |
|---------|-------------------------|--------------------------|
| **E. coli** | All 8 | None |
| **K. pneumoniae** | All 8 | None |
| **P. mirabilis** | 7 (except Imipenem) | Imipenem (predict 1.0) |
| **P. aeruginosa** | 3 (Levo, Cipro, Imipenem) | 5 (predict 1.0) |

### Complexity Reduction

- **Total combinations**: 32
- **Trivial (intrinsic)**: 6
- **Requires ML**: 26
- **Reduction**: 18.8% of predictions can be rule-based

### Recommended Architecture

**Option 1: Hybrid Rule + ML**
```python
def predict(sample):
    species = sample.species_id

    # Apply intrinsic resistance rules
    if species == P_aeruginosa:
        sample[Ampicillin] = 1.0
        sample[Amox_Clav] = 1.0
        sample[Ertapenem] = 1.0
        sample[Cefotaxime] = 1.0
        sample[Cefuroxime] = 1.0
    elif species == P_mirabilis:
        sample[Imipenem] = 1.0

    # Use ML for remaining antibiotics
    remaining = get_non_trivial_antibiotics(species)
    sample[remaining] = ml_model.predict(sample, remaining)
```

**Option 2: Species-Specific Models**
- Train 4 separate models, one per species
- Each model only predicts non-trivial antibiotics
- Reduces complexity and improves specialization

**Option 3: Single Multi-Task Model with Masking**
- Single model with 8 output heads
- Mask out intrinsic resistance combinations during training
- Always predict 1.0 for intrinsic cases at inference

---

## 8.5 MALDI-TOF Biomarker Interpretation

### Expected Biomarker Ranges

MALDI-TOF MS typically identifies biomarkers in:
- **2000-7000 Da**: Most informative for bacterial identification
- **Ribosomal proteins**: Highly abundant, species-specific
- **Housekeeping proteins**: Conserved within species

### Top Discriminative Features

Random Forest analysis identified top features for each antibiotic (see figure: `biomarker_features_by_antibiotic.png`).

### Distribution by m/z Range

- Features in biomarker range (2k-7k Da): 45/50 of top features
- Suggests model is learning biologically relevant patterns
- Validates that MALDI-TOF data is appropriate for AMR prediction

---

## Recommendations

### 1. Immediate Actions

1. **Implement rule-based predictions for intrinsic resistance**:
   ```python
   INTRINSIC_RULES = {
       'P. aeruginosa': ['Ampicillin', 'Amoxicillin_Clavulanic_acid',
                         'Ertapenem', 'Cefotaxime', 'Cefuroxime'],
       'P. mirabilis': ['Imipenem']
   }
   ```

2. **Focus ML modeling on non-trivial combinations**:
   - P. aeruginosa: Only predict Levofloxacin, Ciprofloxacin, Imipenem
   - P. mirabilis: Predict all except Imipenem
   - E. coli & K. pneumoniae: Predict all

3. **Use species-aware architecture**:
   - Include species_id as a feature
   - Or train species-specific models
   - Use species-stratified cross-validation

### 2. Model Architecture

**Recommended**: Hybrid Rule + ML approach
- Deterministic rules for intrinsic resistance (perfect accuracy)
- ML model for remaining combinations (focused learning)
- Significant reduction in model complexity

### 3. Training Strategy

1. **Multi-task learning** for correlated antibiotics
2. **Species-stratified CV** to handle distribution shift
3. **Sample weighting**: Downweight P. aeruginosa (43%→3% shift)
4. **Masked loss**: Ignore intrinsic resistance cases during training

### 4. Evaluation

Track metrics separately:
- Overall mean AUC (competition metric)
- Per-species AUC
- Per-antibiotic AUC
- Distinguish between rule-based and ML-based predictions

---

## Conclusion

This biological analysis reveals significant opportunities for model simplification:

1. **18.8% of predictions can be rule-based** (intrinsic resistance)
2. **Within-class correlations** support multi-task learning
3. **Species-specific patterns** justify specialized modeling
4. **MALDI-TOF features** are biologically interpretable

**Next Steps**:
1. Implement hybrid rule + ML architecture
2. Train species-specific models for non-trivial combinations
3. Leverage antibiotic class correlations in model design
4. Validate that intrinsic resistance rules hold on test set

---

## Generated Files

All figures saved to: `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase8/`

1. `antibiotic_correlation_by_class.png` - Correlation matrix with class groupings
2. `mdr_analysis.png` - MDR prevalence and spectra profiles
3. `intrinsic_resistance_patterns.png` - Resistance rates by species with intrinsic markers
4. `biomarker_features_by_antibiotic.png` - Top RF features per antibiotic
5. `feature_importance_by_mz_range.png` - Feature importance by m/z range
