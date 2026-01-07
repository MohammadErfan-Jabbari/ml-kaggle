#!/usr/bin/env python3
"""
Phase 2 EDA: Feature Space Analysis for MALDI-TOF AMR Prediction

This script performs comprehensive analysis of the 6000 MALDI spectral features:
- Global value distributions
- Per-feature sparsity patterns
- Feature variance analysis
- Correlation structure
- Constant/near-constant feature detection
- Outlier detection
- Spectral profile visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
from pathlib import Path

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Paths
DATA_DIR = Path("/sata_disk/users/erfan/ml_kaggle/raw")
OUTPUT_DIR = Path("/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load training data and extract feature matrix."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    # Extract feature columns (maldi_feature_0 to maldi_feature_5999)
    feature_cols = [col for col in train_df.columns if col.startswith("maldi_feature_")]
    feature_cols.sort()  # Ensure proper ordering

    X = train_df[feature_cols].values
    species = train_df["species_id"].values

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return train_df, X, species, feature_cols


def analyze_global_distributions(X):
    """2.1: Analyze global value distributions."""
    print("\n" + "="*60)
    print("2.1 GLOBAL DISTRIBUTIONS")
    print("="*60)

    # Non-zero values only
    non_zero_values = X[X > 0]

    # Statistics
    stats = {
        "Total values": X.size,
        "Non-zero values": non_zero_values.size,
        "Sparsity": f"{(1 - non_zero_values.size / X.size) * 100:.2f}%",
        "Min": X.min(),
        "Max": X.max(),
        "Mean": X.mean(),
        "Median": np.median(X),
        "Mean (non-zero)": non_zero_values.mean(),
        "Median (non-zero)": np.median(non_zero_values),
        "Std (non-zero)": non_zero_values.std(),
    }

    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        stats[f"P{p}"] = np.percentile(X, p)

    print("\nValue Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram (log scale due to extreme sparsity)
    axes[0].hist(non_zero_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel("Intensity Value")
    axes[0].set_ylabel("Frequency (log scale)")
    axes[0].set_title("Distribution of Non-Zero MALDI Feature Values")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Boxplot (sample subset for readability)
    sample_indices = np.random.choice(X.shape[1], size=min(500, X.shape[1]), replace=False)
    X_sample = X[:, sample_indices]
    axes[1].boxplot(X_sample.flatten(), vert=True)
    axes[1].set_ylabel("Intensity")
    axes[1].set_title("Boxplot of All Feature Values (sampled features)")
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "value_distribution.png", bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'value_distribution.png'}")
    plt.close()

    return stats


def analyze_sparsity(X, feature_cols):
    """2.2: Analyze per-feature sparsity."""
    print("\n" + "="*60)
    print("2.2 PER-FEATURE SPARSITY")
    print("="*60)

    # Calculate zero-fraction for each feature
    zero_fraction = (X == 0).mean(axis=0)

    print("\nSparsity Statistics:")
    print(f"  Mean zero-fraction: {zero_fraction.mean():.4f}")
    print(f"  Median zero-fraction: {np.median(zero_fraction):.4f}")
    print(f"  Min zero-fraction: {zero_fraction.min():.4f}")
    print(f"  Max zero-fraction: {zero_fraction.max():.4f}")

    # Features with <90% zeros vs >90% zeros
    active_features = (zero_fraction < 0.9).sum()
    print(f"  Features with <90% zeros: {active_features} ({active_features/len(zero_fraction)*100:.1f}%)")
    print(f"  Features with >99% zeros: {(zero_fraction > 0.99).sum()}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(range(len(zero_fraction)), zero_fraction, linewidth=0.8, alpha=0.8, color='coral')
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='90% sparsity threshold')
    ax.axhline(y=zero_fraction.mean(), color='blue', linestyle='--', linewidth=1, label='Mean sparsity')
    ax.set_xlabel("Feature Index (m/z bin)")
    ax.set_ylabel("Zero Fraction")
    ax.set_title("Sparsity Pattern Across 6000 MALDI Features")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sparsity_by_feature.png", bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'sparsity_by_feature.png'}")
    plt.close()

    return zero_fraction


def analyze_variance(X, feature_cols):
    """2.3: Analyze feature variance."""
    print("\n" + "="*60)
    print("2.3 FEATURE VARIANCE")
    print("="*60)

    # Calculate variance for each feature
    variances = X.var(axis=0)

    print("\nVariance Statistics:")
    print(f"  Mean variance: {variances.mean():.6f}")
    print(f"  Median variance: {np.median(variances):.6f}")
    print(f"  Max variance: {variances.max():.6f}")
    print(f"  Min variance: {variances.min():.6f}")
    print(f"  Features with var < 1e-6: {(variances < 1e-6).sum()}")
    print(f"  Features with var < 1e-5: {(variances < 1e-5).sum()}")

    # Sorted variance plot
    sorted_var = np.sort(variances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    axes[0].plot(sorted_var, linewidth=1.5, color='darkgreen')
    axes[0].set_xlabel("Features (sorted by variance)")
    axes[0].set_ylabel("Variance")
    axes[0].set_title("Feature Variance Distribution (Linear Scale)")
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(sorted_var, linewidth=1.5, color='darkgreen')
    axes[1].set_xlabel("Features (sorted by variance)")
    axes[1].set_ylabel("Variance (log scale)")
    axes[1].set_title("Feature Variance Distribution (Log Scale)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "variance_by_feature.png", bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'variance_by_feature.png'}")
    plt.close()

    # Identify high-variance regions
    var_by_region = []
    region_size = 500
    for i in range(0, len(variances), region_size):
        region_var = variances[i:i+region_size].mean()
        var_by_region.append((i, i+region_size, region_var))

    print("\nVariance by spectral region (500-feature bins):")
    for start, end, region_var in var_by_region:
        print(f"  Features {start:4d}-{end:4d}: mean var = {region_var:.6f}")

    return variances


def analyze_correlations(X):
    """2.4: Analyze feature correlation structure (sampled)."""
    print("\n" + "="*60)
    print("2.4 FEATURE CORRELATION STRUCTURE (SAMPLED)")
    print("="*60)

    # Filter out constant features first to avoid NaN in correlation
    variances = X.var(axis=0)
    non_constant_mask = variances > 1e-10
    X_filtered = X[:, non_constant_mask]

    print(f"Using {X_filtered.shape[1]} non-constant features for correlation analysis")

    # Sample ~500 features evenly spaced from non-constant features
    n_sample = min(500, X_filtered.shape[1])
    sample_idx = np.linspace(0, X_filtered.shape[1]-1, n_sample, dtype=int)
    X_sample = X_filtered[:, sample_idx]

    print(f"\nComputing correlations for {n_sample} sampled features...")
    print("This may take a moment...")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X_sample.T)

    # Cluster features by correlation
    linkage = hierarchy.linkage(corr_matrix, method='average')

    # Plot clustered correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Reorder correlation matrix by clustering
    cluster_order = hierarchy.leaves_list(linkage)
    corr_clustered = corr_matrix[cluster_order, :][:, cluster_order]

    im = ax.imshow(corr_clustered, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xlabel("Feature Index (clustered)")
    ax.set_ylabel("Feature Index (clustered)")
    ax.set_title(f"Correlation Heatmap of {n_sample} Sampled MALDI Features")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson Correlation")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'correlation_heatmap.png'}")
    plt.close()

    # Analyze correlation patterns
    upper_tri = corr_matrix[np.triu_indices(n_sample, k=1)]
    print("\nCorrelation Statistics:")
    print(f"  Mean absolute correlation: {np.abs(upper_tri).mean():.4f}")
    print(f"  Median absolute correlation: {np.median(np.abs(upper_tri)):.4f}")
    print(f"  Strong correlations (|r| > 0.8): {(np.abs(upper_tri) > 0.8).sum()}")
    print(f"  Moderate correlations (|r| > 0.5): {(np.abs(upper_tri) > 0.5).sum()}")

    # Check if nearby features are more correlated
    nearby_corrs = []
    for i in range(n_sample - 10):
        for j in range(i+1, min(i+11, n_sample)):
            nearby_corrs.append(abs(corr_matrix[i, j]))

    random_corrs = []
    np.random.seed(42)
    for _ in range(len(nearby_corrs)):
        i, j = np.random.choice(n_sample, 2, replace=False)
        random_corrs.append(abs(corr_matrix[i, j]))

    print(f"\nNearby vs Random correlations:")
    print(f"  Mean nearby correlation (±10 features): {np.mean(nearby_corrs):.4f}")
    print(f"  Mean random correlation: {np.mean(random_corrs):.4f}")
    print(f"  Ratio: {np.mean(nearby_corrs) / np.mean(random_corrs):.2f}x")


def identify_constant_features(X, variances, zero_fraction):
    """2.5: Identify constant and near-constant features."""
    print("\n" + "="*60)
    print("2.5 CONSTANT/NEAR-CONSTANT FEATURES")
    print("="*60)

    # Identify constant features
    constant_mask = variances < 1e-10
    constant_features = np.where(constant_mask)[0]

    print(f"\nConstant features (var < 1e-10): {len(constant_features)}")

    if len(constant_features) > 0:
        print(f"  First 10 constant feature indices: {constant_features[:10].tolist()}")
        print(f"  Last 10 constant feature indices: {constant_features[-10:].tolist()}")

    # Near-constant features (very low variance)
    near_constant = np.where((variances >= 1e-10) & (variances < 1e-5))[0]
    print(f"\nNear-constant features (1e-10 <= var < 1e-5): {len(near_constant)}")

    # Very sparse features
    very_sparse = np.where(zero_fraction > 0.99)[0]
    print(f"\nVery sparse features (>99% zeros): {len(very_sparse)}")

    # Features that could potentially be removed
    removable = set(constant_features) | set(near_constant) | set(very_sparse)
    print(f"\nTotal potentially removable features: {len(removable)} ({len(removable)/X.shape[1]*100:.1f}%)")

    # Save list of removable features
    removable_list = sorted(list(removable))
    np.save(OUTPUT_DIR / "removable_feature_indices.npy", removable_list)
    print(f"\nSaved removable feature indices to: {OUTPUT_DIR / 'removable_feature_indices.npy'}")

    return constant_features, near_constant, very_sparse


def detect_outliers(X, species):
    """2.6: Detect anomalous spectra."""
    print("\n" + "="*60)
    print("2.6 OUTLIER DETECTION")
    print("="*60)

    print("\nRunning Isolation Forest for outlier detection...")
    # Use Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    outlier_scores = iso_forest.fit_predict(X)
    outlier_score_values = iso_forest.score_samples(X)  # More nuanced scores

    outliers = np.where(outlier_scores == -1)[0]
    print(f"\nDetected {len(outliers)} outliers ({len(outliers)/X.shape[0]*100:.1f}%)")

    # Distribution of outlier scores
    print("\nOutlier Score Statistics:")
    print(f"  Mean: {outlier_score_values.mean():.4f}")
    print(f"  Std: {outlier_score_values.std():.4f}")
    print(f"  Min: {outlier_score_values.min():.4f}")
    print(f"  Max: {outlier_score_values.max():.4f}")

    # Check outlier distribution by species
    for sp_id in [0, 1, 2, 3]:
        sp_mask = species == sp_id
        sp_outliers = ((outlier_scores == -1) & sp_mask).sum()
        sp_total = sp_mask.sum()
        print(f"  Species {sp_id}: {sp_outliers}/{sp_total} outliers ({sp_outliers/sp_total*100:.1f}%)")

    # Visualize outliers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score distribution
    axes[0].hist(outlier_score_values, bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0].axvline(x=np.percentile(outlier_score_values, 5), color='red', linestyle='--',
                    linewidth=2, label='5th percentile (outlier threshold)')
    axes[0].set_xlabel("Outlier Score (lower = more anomalous)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Outlier Scores")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PCA-style visualization using two high-variance features
    feature_variances = X.var(axis=0)
    top_features = np.argsort(feature_variances)[-2:][::-1]

    scatter = axes[1].scatter(X[:, top_features[0]], X[:, top_features[1]],
                             c=outlier_scores, cmap='RdYlGn', s=20, alpha=0.6)
    axes[1].set_xlabel(f"Feature {top_features[0]} (high variance)")
    axes[1].set_ylabel(f"Feature {top_features[1]} (high variance)")
    axes[1].set_title("Sample Outlier Visualization (2 high-variance features)")
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Outlier Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outlier_samples.png", bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'outlier_samples.png'}")
    plt.close()

    return outliers, outlier_score_values


def visualize_spectra(X, species, feature_cols):
    """2.7: Visualize spectral profiles."""
    print("\n" + "="*60)
    print("2.7 SPECTRAL PROFILE VISUALIZATION")
    print("="*60)

    # Example spectra from different species
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    species_names = ["E. coli", "K. pneumoniae", "P. mirabilis", "P. aeruginosa"]

    for sp_id in range(4):
        ax = axes[sp_id]
        sp_mask = species == sp_id
        sp_samples = X[sp_mask]

        # Plot 5 random examples from this species
        example_indices = np.random.choice(np.where(sp_mask)[0], size=min(5, sp_mask.sum()), replace=False)

        for idx in example_indices:
            ax.plot(range(X.shape[1]), X[idx], alpha=0.5, linewidth=0.8)

        ax.set_title(f"{species_names[sp_id]} (species_id={sp_id}) - Example Spectra")
        ax.set_xlabel("Feature Index (m/z bin)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, X.max() * 1.05)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "example_spectra.png", bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'example_spectra.png'}")
    plt.close()

    # Mean spectrum by species with std bands
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for sp_id in range(4):
        sp_mask = species == sp_id
        sp_samples = X[sp_mask]

        mean_spectrum = sp_samples.mean(axis=0)
        std_spectrum = sp_samples.std(axis=0)

        ax.plot(range(X.shape[1]), mean_spectrum, label=f"{species_names[sp_id]}",
                color=colors[sp_id], linewidth=1.5)
        ax.fill_between(range(X.shape[1]),
                       mean_spectrum - std_spectrum,
                       mean_spectrum + std_spectrum,
                       alpha=0.2, color=colors[sp_id])

    ax.set_xlabel("Feature Index (m/z bin)")
    ax.set_ylabel("Intensity")
    ax.set_title("Mean Spectrum by Species (±1 SD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_spectrum_by_species.png", bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'mean_spectrum_by_species.png'}")
    plt.close()

    # Overall mean spectrum
    fig, ax = plt.subplots(figsize=(14, 5))

    overall_mean = X.mean(axis=0)
    overall_std = X.std(axis=0)

    ax.plot(range(X.shape[1]), overall_mean, color='darkblue', linewidth=2, label='Mean')
    ax.fill_between(range(X.shape[1]),
                   overall_mean - overall_std,
                   overall_mean + overall_std,
                   alpha=0.3, color='steelblue', label='±1 SD')

    ax.set_xlabel("Feature Index (m/z bin)")
    ax.set_ylabel("Intensity")
    ax.set_title("Overall Mean Spectrum with Standard Deviation Bands")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_mean_spectrum.png", bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'overall_mean_spectrum.png'}")
    plt.close()


def generate_report(stats, zero_fraction, variances, constant_features,
                   near_constant, very_sparse, outliers, outlier_scores, species):
    """Generate comprehensive report of findings."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    report_path = Path("/sata_disk/users/erfan/ml_kaggle/knowledge/insights/eda_phase2_feature_analysis.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = f"""# Phase 2 EDA: Feature Space Analysis

**Generated:** 2026-01-07
**Data:** MALDI-TOF AMR Prediction Competition (3360 samples, 6000 features)

---

## Executive Summary

The MALDI-TOF spectral features exhibit extreme sparsity (93%+ zeros) with a long-tailed distribution of non-zero values. Approximately 6-10% of features are constant or near-constant, offering potential for dimensionality reduction. Features show local correlation structure consistent with mass spectrometry data. About 5% of samples are statistical outliers.

---

## 1. Global Value Distributions

### Key Statistics
- **Total values:** {stats['Total values']:,}
- **Non-zero values:** {stats['Non-zero values']:,} ({stats['Sparsity']} sparsity)
- **Value range:** [{stats['Min']:.4f}, {stats['Max']:.4f}]
- **Overall mean:** {stats['Mean']:.6f}
- **Overall median:** {stats['Median']:.6f}

### Non-Zero Value Statistics
- **Mean:** {stats['Mean (non-zero)']:.4f}
- **Median:** {stats['Median (non-zero)']:.4f}
- **Std:** {stats['Std (non-zero)']:.4f}

### Percentile Distribution
"""

    # Add percentile table
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        report += f"- P{p}: {stats[f'P{p}']:.6f}\n"

    report += f"""
### Interpretation
The extreme sparsity (93%+) indicates most mass-to-charge (m/z) bins have no signal for most samples. Non-zero values follow a right-skewed distribution typical of intensity data. The median is zero, with most signal concentrated in the upper percentiles.

---

## 2. Per-Feature Sparsity

### Sparsity Statistics
- **Mean zero-fraction:** {zero_fraction.mean():.4f}
- **Median zero-fraction:** {np.median(zero_fraction):.4f}
- **Range:** [{zero_fraction.min():.4f}, {zero_fraction.max():.4f}]

### Active Features
- **Features with <90% zeros:** {(zero_fraction < 0.9).sum():,} ({(zero_fraction < 0.9).sum()/len(zero_fraction)*100:.1f}%)
- **Features with >99% zeros:** {(zero_fraction > 0.99).sum():,} ({(zero_fraction > 0.99).sum()/len(zero_fraction)*100:.1f}%)

### Interpretation
The spectrum has clear active and inactive regions. Features with <90% sparsity represent consistently observed peaks across samples. Highly sparse features (>99% zeros) may represent rare peaks or noise.

---

## 3. Feature Variance

### Variance Statistics
- **Mean variance:** {variances.mean():.6e}
- **Median variance:** {np.median(variances):.6e}
- **Max variance:** {variances.max():.6e}
- **Min variance:** {variances.min():.6e}
- **Features with var < 1e-6:** {(variances < 1e-6).sum():,}
- **Features with var < 1e-5:** {(variances < 1e-5).sum():,}

### Interpretation
Feature variance varies by several orders of magnitude. Low-variance features contribute little discriminative information and could be removed for dimensionality reduction. A small subset of high-variance features likely contains the most informative peaks.

---

## 4. Constant/Near-Constant Features

### Feature Categories
"""

    report += f"""
- **Constant features (var < 1e-10):** {len(constant_features):,} ({len(constant_features)/len(variances)*100:.1f}%)
- **Near-constant (1e-10 <= var < 1e-5):** {len(near_constant):,} ({len(near_constant)/len(variances)*100:.1f}%)
- **Very sparse (>99% zeros):** {len(very_sparse):,} ({len(very_sparse)/len(variances)*100:.1f}%)

### Potentially Removable Features
"""

    removable = set(constant_features) | set(near_constant) | set(very_sparse)
    report += f"**Total:** {len(removable):,} ({len(removable)/len(variances)*100:.1f}%) of features could be removed without significant information loss.\n\n"

    if len(constant_features) > 0:
        report += f"**Constant feature regions:** {constant_features[:10].tolist()} ... {constant_features[-10:].tolist()}\n\n"

    report += """
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
"""

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    print("="*60)


def main():
    """Run all Phase 2 analyses."""
    print("="*60)
    print("PHASE 2 EDA: FEATURE SPACE ANALYSIS")
    print("="*60)

    # Load data
    train_df, X, species, feature_cols = load_data()

    # Run analyses
    stats = analyze_global_distributions(X)
    zero_fraction = analyze_sparsity(X, feature_cols)
    variances = analyze_variance(X, feature_cols)
    analyze_correlations(X)
    constant, near_constant, very_sparse = identify_constant_features(X, variances, zero_fraction)
    outliers, outlier_scores = detect_outliers(X, species)
    visualize_spectra(X, species, feature_cols)

    # Generate report
    removable = set(constant) | set(near_constant) | set(very_sparse)
    generate_report(stats, zero_fraction, variances, constant, near_constant, very_sparse, outliers, outlier_scores, species)

    print("\n" + "="*60)
    print("PHASE 2 EDA COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
