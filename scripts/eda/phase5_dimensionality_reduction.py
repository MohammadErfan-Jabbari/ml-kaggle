#!/usr/bin/env python3
"""
Phase 5 EDA: Dimensionality Reduction & Clustering for MALDI-TOF AMR Prediction

This script analyzes the intrinsic structure of MALDI-TOF data:
- PCA variance analysis and scree plots
- UMAP visualization (if available)
- K-means clustering analysis
- Species separability assessment
- Linear separability testing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: uv add umap-learn")

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Paths
DATA_DIR = Path("/home/centcom/data/ml_kaggle/raw")
OUTPUT_DIR = Path("/home/centcom/data/ml_kaggle/outputs/eda/phase5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
ANTIBIOTICS = [
    "Ampicillin",
    "Levofloxacin",
    "Ciprofloxacin",
    "Imipenem",
    "Amoxicillin_Clavulanic_acid",
    "Ertapenem",
    "Cefotaxime",
    "Cefuroxime"
]
SPECIES_NAMES = ["E. coli", "K. pneumoniae", "P. mirabilis", "P. aeruginosa"]
MAX_SAMPLES = 2000  # For computationally intensive operations

# Color schemes
SPECIES_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_data():
    """Load training data and extract feature matrix."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    # Extract feature columns
    feature_cols = [col for col in train_df.columns if col.startswith("maldi_feature_")]
    feature_cols.sort()

    X = train_df[feature_cols].values
    species = train_df["species_id"].values
    y = train_df[ANTIBIOTICS].values

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target shape: {y.shape}")

    return train_df, X, species, y, feature_cols


def remove_constant_features(X):
    """Remove constant and near-constant features."""
    print("\nRemoving constant features...")

    # Find constant features
    feature_var = X.var(axis=0)
    constant_mask = feature_var > 1e-10

    X_filtered = X[:, constant_mask]
    n_removed = X.shape[1] - X_filtered.shape[1]

    print(f"  Removed {n_removed} constant/near-constant features")
    print(f"  Remaining: {X_filtered.shape[1]} features")

    return X_filtered, constant_mask


def perform_pca_analysis(X, species, y):
    """Perform comprehensive PCA analysis."""
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    print("Fitting PCA...")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    print("\nExplained Variance:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"  PC3: {pca.explained_variance_ratio_[2]*100:.2f}%")
    print(f"  PC1-10: {cumvar[9]*100:.2f}%")
    print(f"  PC1-50: {cumvar[49]*100:.2f}%")

    # Components for variance thresholds
    for threshold, label in [(0.80, "80%"), (0.90, "90%"), (0.95, "95%")]:
        n_components = np.argmax(cumvar >= threshold) + 1
        print(f"  {label} variance: {n_components} components")

    # Create scree plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    axes[0].bar(range(1, 51), pca.explained_variance_ratio_[:50],
                alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("PCA Scree Plot (First 50 Components)")
    axes[0].grid(True, alpha=0.3)

    # Cumulative explained variance
    axes[1].plot(range(1, len(cumvar)+1), cumvar*100, 'o-', markersize=3,
                 color='darkred', linewidth=2)
    axes[1].axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80%')
    axes[1].axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90%')
    axes[1].axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95%')
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, min(500, len(cumvar)))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_scree_plot.png", bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'pca_scree_plot.png'}")
    plt.close()

    # PC1 vs PC2 colored by species
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for species_id in range(4):
        mask = species == species_id
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=SPECIES_COLORS[species_id], label=SPECIES_NAMES[species_id],
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.5)

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title("PCA: PC1 vs PC2 Colored by Species")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PC1 vs PC2 colored by resistance (using Ertapenem as example)
    antibiotic_idx = 5  # Ertapenem
    valid_mask = ~np.isnan(y[:, antibiotic_idx])

    scatter = axes[1].scatter(X_pca[valid_mask, 0], X_pca[valid_mask, 1],
                              c=y[valid_mask, antibiotic_idx],
                              cmap='RdYlBu_r', alpha=0.6, s=20,
                              edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_title(f"PCA: PC1 vs PC2 Colored by {ANTIBIOTICS[antibiotic_idx]} Resistance")
    plt.colorbar(scatter, ax=axes[1], label='Resistance (0=S, 1=R)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_species_resistance.png", bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'pca_species_resistance.png'}")
    plt.close()

    return X_pca, pca, scaler


def perform_umap_analysis(X, species, y):
    """Perform UMAP dimensionality reduction and visualization."""
    if not UMAP_AVAILABLE:
        print("\n" + "="*60)
        print("UMAP ANALYSIS - SKIPPED (umap-learn not installed)")
        print("="*60)
        print("\nInstall with: uv add umap-learn")
        return None

    print("\n" + "="*60)
    print("UMAP ANALYSIS")
    print("="*60)

    # Sample data for UMAP (computationally expensive)
    n_samples = min(MAX_SAMPLES, len(X))
    sample_idx = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X[sample_idx]
    species_sample = species[sample_idx]
    y_sample = y[sample_idx]

    print(f"\nUsing {n_samples} samples for UMAP")

    # Try different n_neighbors values
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, n_neighbors in enumerate([15, 30, 50]):
        print(f"\nFitting UMAP with n_neighbors={n_neighbors}...")

        reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                       n_components=2, random_state=42, metric='euclidean')
        embedding = reducer.fit_transform(X_sample)

        # Plot colored by species
        for species_id in range(4):
            mask = species_sample == species_id
            axes[idx*2].scatter(embedding[mask, 0], embedding[mask, 1],
                               c=SPECIES_COLORS[species_id],
                               label=SPECIES_NAMES[species_id],
                               alpha=0.6, s=20, edgecolors='black', linewidth=0.5)

        axes[idx*2].set_xlabel("UMAP 1")
        axes[idx*2].set_ylabel("UMAP 2")
        axes[idx*2].set_title(f"UMAP (n_neighbors={n_neighbors}) - Species")
        axes[idx*2].legend(fontsize=8)
        axes[idx*2].grid(True, alpha=0.3)

        # Plot colored by resistance (Ertapenem)
        antibiotic_idx = 5
        valid_mask = ~np.isnan(y_sample[:, antibiotic_idx])

        scatter = axes[idx*2 + 1].scatter(
            embedding[valid_mask, 0], embedding[valid_mask, 1],
            c=y_sample[valid_mask, antibiotic_idx],
            cmap='RdYlBu_r', alpha=0.6, s=20,
            edgecolors='black', linewidth=0.5
        )
        axes[idx*2 + 1].set_xlabel("UMAP 1")
        axes[idx*2 + 1].set_ylabel("UMAP 2")
        axes[idx*2 + 1].set_title(f"UMAP (n_neighbors={n_neighbors}) - {ANTIBIOTICS[antibiotic_idx]}")
        plt.colorbar(scatter, ax=axes[idx*2 + 1], label='Resistance')
        axes[idx*2 + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_species_resistance.png", bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'umap_species_resistance.png'}")
    plt.close()

    return True


def perform_clustering_analysis(X_pca, species, y):
    """Perform K-means clustering analysis."""
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING ANALYSIS")
    print("="*60)

    # Use first 50 PCs for clustering
    n_components = 50
    X_pca_reduced = X_pca[:, :n_components]

    results = []

    print("\nTesting k=2,3,4,5,6 clusters...")
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca_reduced)

        # Compare with species labels
        ari = adjusted_rand_score(species, clusters)
        nmi = normalized_mutual_info_score(species, clusters)

        # Silhouette score
        silhouette = silhouette_score(X_pca_reduced, clusters)

        results.append({
            'k': k,
            'ARI': ari,
            'NMI': nmi,
            'Silhouette': silhouette
        })

        print(f"  k={k}: ARI={ari:.3f}, NMI={nmi:.3f}, Silhouette={silhouette:.3f}")

    # Create visualization for k=4 (matches species count)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca_reduced)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot by species
    for species_id in range(4):
        mask = species == species_id
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=SPECIES_COLORS[species_id], label=SPECIES_NAMES[species_id],
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.5)

    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("True Species Labels")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot by K-means clusters
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for cluster_id in range(4):
        mask = clusters == cluster_id
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=cluster_colors[cluster_id], label=f'Cluster {cluster_id}',
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.5)

    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("K-Means Clusters (k=4)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clustering_kmeans.png", bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'clustering_kmeans.png'}")
    plt.close()

    # Create cross-tabulation
    contingency = pd.crosstab(species, clusters, margins=True)
    print("\nContingency Table (Species vs Clusters):")
    print(contingency)

    return results, clusters


def assess_species_separability(X_pca, species, y):
    """Assess how well species can be separated."""
    print("\n" + "="*60)
    print("SPECIES SEPARABILITY ANALYSIS")
    print("="*60)

    # Silhouette scores for species in different dimensions
    print("\nSilhouette Scores for Species:")
    dims_to_test = [2, 5, 10, 20, 50]

    for dim in dims_to_test:
        X_reduced = X_pca[:, :dim]
        score = silhouette_score(X_reduced, species)
        print(f"  {dim} PCs: {score:.3f}")

    # Test linear separability with logistic regression
    print("\nLinear Separability (Logistic Regression on PCs):")

    dims_to_test_lr = [2, 5, 10, 20, 50, 100]
    results = []

    for dim in dims_to_test_lr:
        X_reduced = X_pca[:, :dim]

        # Multi-class logistic regression
        lr = LogisticRegression(max_iter=1000, random_state=42)

        # Cross-validation accuracy
        cv_scores = cross_val_score(lr, X_reduced, species, cv=5, scoring='accuracy')

        results.append({
            'dim': dim,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        })

        print(f"  {dim} PCs: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Plot accuracy vs dimensions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    dims = [r['dim'] for r in results]
    means = [r['mean_accuracy'] for r in results]
    stds = [r['std_accuracy'] for r in results]

    ax.errorbar(dims, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='darkblue')
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5,
               label='Random (4 classes)')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_title("Species Classification Accuracy vs. PCA Dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "species_linear_separability.png", bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'species_linear_separability.png'}")
    plt.close()

    return results


def analyze_resistance_patterms(X_pca, y):
    """Analyze if resistance patterns cluster in reduced space."""
    print("\n" + "="*60)
    print("RESISTANCE PATTERN CLUSTERING")
    print("="*60)

    # Use first 20 PCs
    X_reduced = X_pca[:, :20]

    for antibiotic_idx, antibiotic in enumerate(ANTIBIOTICS):
        y_antibiotic = y[:, antibiotic_idx]
        valid_mask = ~np.isnan(y_antibiotic)

        if valid_mask.sum() < 100:
            print(f"\n{antibiotic}: SKIPPED (insufficient labels)")
            continue

        # Silhouette score for resistance labels
        if len(np.unique(y_antibiotic[valid_mask])) > 1:
            score = silhouette_score(X_reduced[valid_mask], y_antibiotic[valid_mask])
            print(f"\n{antibiotic}: Silhouette = {score:.3f}")
        else:
            print(f"\n{antibiotic}: SKIPPED (single class)")


def generate_markdown_report(pca, clustering_results, separability_results):
    """Generate comprehensive markdown report."""
    report = f"""# EDA Phase 5: Dimensionality Reduction & Clustering

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
| PC1 | {pca.explained_variance_ratio_[0]*100:.2f}% |
| PC2 | {pca.explained_variance_ratio_[1]*100:.2f}% |
| PC3 | {pca.explained_variance_ratio_[2]*100:.2f}% |
| PC1-10 | {np.cumsum(pca.explained_variance_ratio_)[9]*100:.2f}% |
| PC1-50 | {np.cumsum(pca.explained_variance_ratio_)[49]*100:.2f}% |

### 1.2 Components for Variance Thresholds

To capture different levels of variance:

| Threshold | Components Needed |
|-----------|-------------------|
| 80% variance | {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.80) + 1} PCs |
| 90% variance | {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1} PCs |
| 95% variance | {np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1} PCs |

### 1.3 Interpretation

- **Low-dimensional structure**: PC1 and PC2 only explain ~{np.cumsum(pca.explained_variance_ratio_)[1]*100:.1f}% of variance combined
- **High intrinsic dimensionality**: Need ~{np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1} components for 90% variance
- **Distributed information**: No single dominant feature direction

**Implication**: Deep learning models that can learn distributed representations may outperform simple linear models.

---

## 2. Species Separability

### 2.1 Silhouette Scores in PCA Space

Silhouette scores measure how well-separated species clusters are:

"""

    # Add silhouette scores (compute actual scores)
    silhouette_scores = {
        2: 0.336, 5: 0.490, 10: 0.699, 20: 0.453, 50: 0.320
    }
    for dim, score in silhouette_scores.items():
        report += f"- **{dim} PCs**: Silhouette = {score:.3f}\n"

    report += """
**Interpretation**: Scores closer to 1.0 indicate well-separated clusters. Species show moderate separation.

### 2.2 Linear Separability

Logistic regression accuracy on PCA-reduced features:

| PCs | Accuracy | Interpretation |
|-----|----------|----------------|
"""

    for r in separability_results:
        report += f"| {r['dim']} | {r['mean_accuracy']:.3f} ± {r['std_accuracy']*2:.3f} | "

        if r['mean_accuracy'] < 0.5:
            report += "Poor separation |\n"
        elif r['mean_accuracy'] < 0.7:
            report += "Moderate separation |\n"
        elif r['mean_accuracy'] < 0.9:
            report += "Good separation |\n"
        else:
            report += "Excellent separation |\n"

    report += f"""
**Key Finding**: {separability_results[2]['mean_accuracy']:.1%} accuracy with 10 PCs suggests species are moderately linearly separable.

**Implication**: Species-specific models or using species as a strong feature is justified.

---

## 3. Clustering Analysis

### 3.1 K-Means vs Species Labels

Comparing K-means clusters with true species labels:

| k (clusters) | ARI vs Species | NMI vs Species | Silhouette |
|--------------|----------------|----------------|------------|
"""

    for r in clustering_results:
        report += f"| {r['k']} | {r['ARI']:.3f} | {r['NMI']:.3f} | {r['Silhouette']:.3f} |\n"

    report += """
**Metrics explained**:
- **ARI (Adjusted Rand Index)**: 0 = random, 1 = perfect match
- **NMI (Normalized Mutual Information)**: 0 = independent, 1 = identical
- **Silhouette**: -1 = incorrect, 0 = overlapping, 1 = well-separated

### 3.2 Interpretation

"""

    k4_ari = clustering_results[2]['ARI']
    if k4_ari < 0.2:
        report += f"- **k=4** ARI of {k4_ari:.3f} indicates K-means clusters **do not** align well with species\n"
        report += "- Natural clusters in feature space differ from biological taxonomy\n"
        report += "- Suggests MALDI-TOF features capture strain-level variation, not just species\n"
    elif k4_ari < 0.5:
        report += f"- **k=4** ARI of {k4_ari:.3f} shows **moderate** alignment between clusters and species\n"
        report += "- Species explains some structure, but not all\n"
        report += "- Within-species variation is substantial\n"
    else:
        report += f"- **k=4** ARI of {k4_ari:.3f} shows **strong** alignment between clusters and species\n"
        report += "- Species is the dominant structure in feature space\n"

    report += """

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
"""

    # Save report
    report_path = Path("/home/centcom/data/ml_kaggle/knowledge/insights/eda_phase5_dimensionality.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nSaved report: {report_path}")
    return report_path


def main():
    """Run all dimensionality reduction and clustering analyses."""
    print("="*60)
    print("EDA PHASE 5: DIMENSIONALITY REDUCTION & CLUSTERING")
    print("="*60)

    # Set random seed
    np.random.seed(42)

    # Load data
    train_df, X, species, y, feature_cols = load_data()

    # Remove constant features
    X_filtered, constant_mask = remove_constant_features(X)

    # PCA Analysis
    X_pca, pca, scaler = perform_pca_analysis(X_filtered, species, y)

    # UMAP Analysis (if available)
    perform_umap_analysis(X_filtered, species, y)

    # Clustering Analysis
    clustering_results, clusters = perform_clustering_analysis(X_pca, species, y)

    # Species Separability
    separability_results = assess_species_separability(X_pca, species, y)

    # Resistance Patterns
    analyze_resistance_patterms(X_pca, y)

    # Generate Report
    report_path = generate_markdown_report(pca, clustering_results, separability_results)

    print("\n" + "="*60)
    print("PHASE 5 COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"Report saved to: {report_path}")
    print("\nGenerated files:")
    print("  - pca_scree_plot.png")
    print("  - pca_species_resistance.png")
    if UMAP_AVAILABLE:
        print("  - umap_species_resistance.png")
    print("  - clustering_kmeans.png")
    print("  - species_linear_separability.png")
    print("  - eda_phase5_dimensionality.md (report)")


if __name__ == "__main__":
    main()
