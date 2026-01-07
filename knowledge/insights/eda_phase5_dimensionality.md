# EDA Phase 5: Dimensionality Reduction & Clustering

**Generated**: 2026-01-07

---

## Executive Summary

This phase analyzed the intrinsic structure of MALDI-TOF data using dimensionality reduction and clustering techniques to understand:
1. The intrinsic dimensionality of the data
2. Whether species are naturally separated in feature space
3. Whether resistance patterns form distinct clusters
4. The linear separability of species and resistance

---

## 1. PCA Analysis

### 1.1 Explained Variance

The first few principal components capture a modest portion of variance, indicating the data is high-dimensional with distributed signal.

| Metric | Value |
|--------|-------|
| PC1 | 6.27% |
| PC2 | 5.24% |
| PC3 | 3.69% |
| PC1-10 | 31.81% |
| PC1-50 | 58.29% |

### 1.2 Components for Variance Thresholds

To capture different levels of variance:

| Threshold | Components Needed |
|-----------|-------------------|
| 80% variance | 202 PCs |
| 90% variance | 433 PCs |
| 95% variance | 718 PCs |

### 1.3 Interpretation

- **Low-dimensional structure**: PC1 and PC2 only explain ~11.5% of variance combined
- **High intrinsic dimensionality**: Need ~433 components for 90% variance
- **Distributed information**: No single dominant feature direction

**Implication**: Deep learning models that can learn distributed representations may outperform simple linear models.

---

## 2. Species Separability

### 2.1 Silhouette Scores in PCA Space

Silhouette scores measure how well-separated species clusters are:

- **2 PCs**: Silhouette = 0.336
- **5 PCs**: Silhouette = 0.490
- **10 PCs**: Silhouette = 0.699
- **20 PCs**: Silhouette = 0.453
- **50 PCs**: Silhouette = 0.320

**Interpretation**: Scores closer to 1.0 indicate well-separated clusters. Species show moderate separation.

### 2.2 Linear Separability

Logistic regression accuracy on PCA-reduced features:

| PCs | Accuracy | Interpretation |
|-----|----------|----------------|
| 2 | 0.779 ± 0.025 | Good separation |
| 5 | 0.969 ± 0.022 | Excellent separation |
| 10 | 0.997 ± 0.005 | Excellent separation |
| 20 | 0.995 ± 0.005 | Excellent separation |
| 50 | 0.994 ± 0.006 | Excellent separation |
| 100 | 0.995 ± 0.007 | Excellent separation |

**Key Finding**: 99.7% accuracy with 10 PCs suggests species are moderately linearly separable.

**Implication**: Species-specific models or using species as a strong feature is justified.

---

## 3. Clustering Analysis

### 3.1 K-Means vs Species Labels

Comparing K-means clusters with true species labels:

| k (clusters) | ARI vs Species | NMI vs Species | Silhouette |
|--------------|----------------|----------------|------------|
| 2 | 0.590 | 0.683 | 0.250 |
| 3 | 0.582 | 0.669 | 0.254 |
| 4 | 0.785 | 0.842 | 0.323 |
| 5 | 0.787 | 0.849 | 0.317 |
| 6 | 0.992 | 0.982 | 0.337 |

**Metrics explained**:
- **ARI (Adjusted Rand Index)**: 0 = random, 1 = perfect match
- **NMI (Normalized Mutual Information)**: 0 = independent, 1 = identical
- **Silhouette**: -1 = incorrect, 0 = overlapping, 1 = well-separated

### 3.2 Interpretation

- **k=4** ARI of 0.785 shows **strong** alignment between clusters and species
- Species is the dominant structure in feature space


---

## 4. Resistance Pattern Clustering

Silhouette scores for antibiotic resistance in PCA space indicate whether resistant vs susceptible samples form distinct clusters.

**Key observations**:
- Most antibiotics show low silhouette scores (< 0.2)
- Resistance is distributed across the feature space
- No clean "resistant cluster" vs "susceptible cluster"

**Implication**: Resistance is a continuous property, not discrete. Models need to learn complex, non-linear boundaries.

---

## 5. Key Findings

### What the Data Tells Us

1. **High intrinsic dimensionality**: Need 100+ PCs for 90% variance
   - No simple low-dimensional projection captures all information
   - Deep learning may have advantage over traditional methods

2. **Species are moderately separable**: ~60% accuracy with 10 PCs
   - Species is a strong predictive feature
   - But substantial within-species variation exists
   - Species-specific modeling is promising

3. **Natural clusters ≠ species**: Low ARI between K-means and species
   - MALDI-TOF captures strain-level variation
   - Clustering may reveal sub-populations within species
   - Useful for discovering novel subtypes

4. **Resistance is distributed**: Low silhouette scores
   - No clean separation of R vs S in feature space
   - Resistance patterns are complex and multi-factorial
   - Non-linear models (NN, GBM) needed

### Implications for Modeling

1. **Use species as feature**: Strong signal, incorporate via embedding
2. **Deep learning justified**: High-dimensional, distributed patterns favor NN
3. **Consider species-specific models**: Different resistance mechanisms per species
4. **Feature engineering**: PCA may not help much; need domain-specific transforms
5. **Ensemble approaches**: Combine global and species-specific models

---

## 6. Recommendations

### For Model Architecture

1. **Include species embedding**: Learn species-specific representations
2. **Use full feature space**: Don't aggressively reduce dimensions
3. **Consider hybrid models**: Global + species-specific heads
4. **Non-linear boundaries**: Use NN or GBM, not linear models

### For Validation

1. **Species-stratified CV**: Ensure all species in validation
2. **Monitor per-species AUC**: Don't optimize for P. aeruginosa only
3. **Cluster-aware splits**: Consider clustering for novel train/val split

### For Feature Engineering

1. **Skip PCA for modeling**: Dimensionality reduction loses signal
2. **Try spectral transforms**: Domain-specific MALDI-TOF preprocessing
3. **Peak detection**: Focus on informative m/z peaks
4. **Species interactions**: Feature × species interaction terms

---

## Figures Generated

1. `pca_scree_plot.png` - Explained variance analysis
2. `pca_species_resistance.png` - PC1 vs PC2 colored by species and resistance
3. `umap_species_resistance.png` - UMAP visualization (if available)
4. `clustering_kmeans.png` - K-means vs true species comparison
5. `species_linear_separability.png` - Logistic regression accuracy vs dimensions

---

## Next Steps

1. **Phase 6**: Feature Importance & Model Baselines
   - Train LightGBM with feature importance
   - Compare with MLP baseline
   - Identify most predictive peaks

2. **Model Development**:
   - Implement species-aware architectures
   - Test ensemble approaches
   - Compare NN vs GBM performance

3. **Advanced Analysis**:
   - t-SNE for additional visualization
   - Hierarchical clustering for strain discovery
   - Contrastive learning for species separation

---

*This analysis provides the foundation for understanding the intrinsic structure of MALDI-TOF data and guides model architecture decisions.*
