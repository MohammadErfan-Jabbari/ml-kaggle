# Phase 2 EDA: Feature Space Analysis

**Generated:** 2026-01-07
**Data:** MALDI-TOF AMR Prediction Competition (3360 samples, 6000 features)

---

## Executive Summary

The MALDI-TOF spectral features exhibit extreme sparsity (93%+ zeros) with a long-tailed distribution of non-zero values. Approximately 6-10% of features are constant or near-constant, offering potential for dimensionality reduction. Features show local correlation structure consistent with mass spectrometry data. About 5% of samples are statistical outliers.

---

## 1. Global Value Distributions

### Key Statistics
- **Total values:** 20,160,000
- **Non-zero values:** 1,339,892 (93.35% sparsity)
- **Value range:** [0.0000, 3.0115]
- **Overall mean:** 0.086898
- **Overall median:** 0.000000

### Non-Zero Value Statistics
- **Mean:** 1.3075
- **Median:** 1.3084
- **Std:** 0.4230

### Percentile Distribution
- P1: 0.000000
- P5: 0.000000
- P10: 0.000000
- P25: 0.000000
- P50: 0.000000
- P75: 0.000000
- P90: 0.000000
- P95: 1.038223
- P99: 1.735173

### Interpretation
The extreme sparsity (93%+) indicates most mass-to-charge (m/z) bins have no signal for most samples. Non-zero values follow a right-skewed distribution typical of intensity data. The median is zero, with most signal concentrated in the upper percentiles.

---

## 2. Per-Feature Sparsity

### Sparsity Statistics
- **Mean zero-fraction:** 0.9335
- **Median zero-fraction:** 0.9949
- **Range:** [0.1982, 1.0000]

### Active Features
- **Features with <90% zeros:** 1,223 (20.4%)
- **Features with >99% zeros:** 3,307 (55.1%)

### Interpretation
The spectrum has clear active and inactive regions. Features with <90% sparsity represent consistently observed peaks across samples. Highly sparse features (>99% zeros) may represent rare peaks or noise.

---

## 3. Feature Variance

### Variance Statistics
- **Mean variance:** 8.630946e-02
- **Median variance:** 4.476914e-03
- **Max variance:** 1.261165e+00
- **Min variance:** 0.000000e+00
- **Features with var < 1e-6:** 373
- **Features with var < 1e-5:** 400

### Interpretation
Feature variance varies by several orders of magnitude. Low-variance features contribute little discriminative information and could be removed for dimensionality reduction. A small subset of high-variance features likely contains the most informative peaks.

---

## 4. Constant/Near-Constant Features

### Feature Categories

- **Constant features (var < 1e-10):** 365 (6.1%)
- **Near-constant (1e-10 <= var < 1e-5):** 35 (0.6%)
- **Very sparse (>99% zeros):** 3,307 (55.1%)

### Potentially Removable Features
**Total:** 3,307 (55.1%) of features could be removed without significant information loss.

**Constant feature regions:** [2069, 2070, 2346, 2477, 2628, 2944, 3137, 3138, 3275, 3506] ... [5515, 5516, 5517, 5536, 5537, 5547, 5551, 5552, 5553, 5554]


### Interpretation
These constant features provide no discriminative power and can be safely removed to reduce dimensionality from 6000 to ~5400-5600 features. This will:
- Reduce memory footprint
- Speed up training
- Reduce overfitting risk
- Improve model generalization

---

## 5. Correlation Structure

### Key Findings
- **Nearby features** (within ±10 indices) show systematically higher correlations than random feature pairs
- **Correlation magnitude** is modest (typically <0.5), indicating diverse spectral information
- **Cluster structure** suggests grouped peaks that may represent related molecular species

### Interpretation
The local correlation structure is expected for binned mass spectrometry data—nearby m/z bins represent similar masses. This structure can be exploited for:
- Feature grouping
- Dimensionality reduction (PCA, autoencoders)
- Convolutional neural network architectures

---

## 6. Outlier Detection

### Outlier Statistics
- **Detected outliers:** 168 (5.0% of samples)
- **Outlier rate by species:**
  - E. coli (0): 39/559 (7.0%)
  - K. pneumoniae (1): 26/939 (2.8%)
  - P. mirabilis (2): 50/415 (12.0%)
  - P. aeruginosa (3): 53/1447 (3.7%)
- **Method:** Isolation Forest (contamination=5%)

### Interpretation
Outlier spectra may represent:
- Measurement artifacts or instrument errors
- Rare species variants or mixed cultures
- Sample preparation issues
- Legitimate biological extremes (e.g., hyper-resistant strains)

### Recommendations
- Investigate outlier samples for quality issues
- Consider robust modeling techniques less sensitive to outliers
- Do not automatically remove outliers—they may contain important resistance patterns

---

## 7. Spectral Profiles by Species

### Key Observations
- Each species shows **distinct spectral signatures**
- Mean spectra vary significantly between species
- Within-species variability is substantial (see std bands in visualizations)
- **P. aeruginosa** (species_id=3) dominates training data but shows unique profile

### Interpretation
Species is a major source of spectral variation. This explains the strong predictive power of species_id for resistance. However, within-species spectral variation still contains important resistance-related information.

---

## Implications for Modeling

### Preprocessing Recommendations

1. **Feature Selection**
   - Remove 400 constant/near-constant features
   - Consider variance thresholding (e.g., var > 1e-5)
   - Keep ~5600 informative features

2. **Normalization**
   - Handle extreme sparsity explicitly
   - Consider log1p transform: `log(1 + X)` to compress dynamic range
   - StandardScaler after sparsity handling
   - Alternative: Max scaling per sample (total ion current normalization)

3. **Dimensionality Reduction**
   - PCA: Exploit correlation structure
   - Feature clustering: Group correlated m/z bins
   - Autoencoders: Learn compressed spectral representations

4. **Architecture Considerations**
   - **CNNs:** 1D convolutions can exploit local correlation structure
   - **Tree-based models:** Handle sparsity well, no scaling needed
   - **Attention mechanisms:** Identify informative spectral regions
   - **Species embeddings:** Explicitly model species-specific patterns

### Feature Engineering Opportunities

1. **Peak presence indicators:** Binary features for peaks above threshold
2. **Spectral regions:** Aggregate statistics over m/z ranges
3. **Species-specific features:** Interactions between spectral features and species
4. **Peak ratios:** Relative intensities of neighboring peaks
5. **Total ion current:** Per-sample normalization factor

### Data Quality Actions

1. **Investigate outliers:** Verify measurement quality
2. **Review sparse features:** Distinguish signal from noise
3. **Validate species labels:** Check consistency with spectral profiles
4. **Consider sample weighting:** Down-weight P. aeruginosa to match test distribution

---

## Next Steps

1. **Implement preprocessing pipeline** with feature selection
2. **Compare tree-based vs neural models** on preprocessed features
3. **Explore species-stratified modeling** (separate models per species)
4. **Investigate outlier samples** for data quality issues
5. **Test dimensionality reduction** (PCA, UMAP for visualization)

---

## Files Generated

All figures saved to: `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase2/`

- `value_distribution.png` - Global value distributions
- `sparsity_by_feature.png` - Sparsity across 6000 features
- `variance_by_feature.png` - Variance distribution (linear and log scale)
- `correlation_heatmap.png` - Clustered correlations (500 sampled features)
- `example_spectra.png` - Example spectra from each species
- `mean_spectrum_by_species.png` - Mean ± SD by species
- `overall_mean_spectrum.png` - Global mean spectrum
- `outlier_samples.png` - Outlier detection visualization

**Removable feature indices:** `outputs/eda/phase2/removable_feature_indices.npy`
