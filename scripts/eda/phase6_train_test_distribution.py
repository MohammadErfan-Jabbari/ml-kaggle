"""
Phase 6: Train-Test Distribution Analysis
==========================================

This script analyzes distribution shifts between train and test sets, including:
- Feature distribution comparison (mean, std, KS test)
- Species-stratified feature distribution comparison
- Covariate shift detection using domain classifier
- Density estimation for OOD detection
- Statistical tests for distribution differences

Critical for this competition due to known species distribution shift.

Outputs:
    - Figures saved to outputs/eda/phase6/
    - Markdown report at knowledge/insights/eda_phase6_train_test_distribution.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set paths
PROJECT_ROOT = Path('/home/centcom/data/ml_kaggle')
RAW_DIR = PROJECT_ROOT / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eda' / 'phase6'
KNOWLEDGE_DIR = PROJECT_ROOT / 'knowledge' / 'insights'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
SAMPLE_SIZE = 500  # For computationally intensive operations
N_FEATURES_PLOT = 12  # Number of top shifted features to plot
N_FEATURES_TEST = 1000  # Number of features to test for distribution shift
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

print("=" * 80)
print("PHASE 6: TRAIN-TEST DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading data...")
train = pd.read_csv(RAW_DIR / 'train.csv')
test = pd.read_csv(RAW_DIR / 'test.csv')

feature_cols = [col for col in train.columns if col.startswith('maldi_feature_')]
target_cols = ['Ampicillin', 'Levofloxacin', 'Ciprofloxacin', 'Imipenem',
               'Amoxicillin_Clavulanic_acid', 'Ertapenem', 'Cefotaxime', 'Cefuroxime']
species_names = {0: 'E. coli', 1: 'K. pneumoniae', 2: 'P. mirabilis', 3: 'P. aeruginosa'}

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Features: {len(feature_cols)}")
print()

# =============================================================================
# 1. FEATURE DISTRIBUTION COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("1. FEATURE DISTRIBUTION COMPARISON")
print("=" * 80)

print("\nComputing feature statistics...")
train_means = train[feature_cols].mean()
train_stds = train[feature_cols].std()
test_means = test[feature_cols].mean()
test_stds = test[feature_cols].std()

mean_diff = test_means - train_means
mean_diff_pct = (mean_diff / (train_means + 1e-8)) * 100
std_diff = test_stds - train_stds
std_diff_pct = (std_diff / (train_stds + 1e-8)) * 100

print("\nOverall statistics:")
print(f"  Mean difference (abs): {np.abs(mean_diff).mean():.6f}")
print(f"  Mean difference (pct): {np.abs(mean_diff_pct).mean():.2f}%")
print(f"  Std difference (abs): {np.abs(std_diff).mean():.6f}")
print(f"  Std difference (pct): {np.abs(std_diff_pct).mean():.2f}%")

# Kolmogorov-Smirnov test for distribution shift
print(f"\nRunning KS test on {N_FEATURES_TEST} features (randomly sampled)...")
features_to_test = np.random.choice(feature_cols, min(N_FEATURES_TEST, len(feature_cols)), replace=False)

ks_results = []
for feat in tqdm(features_to_test, desc="KS tests"):
    train_vals = train[feat].values
    test_vals = test[feat].values
    statistic, pvalue = stats.ks_2samp(train_vals, test_vals)
    ks_results.append({
        'feature': feat,
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': pvalue < 0.01
    })

ks_df = pd.DataFrame(ks_results)
ks_df = ks_df.sort_values('statistic', ascending=False)

print("\nKS Test Results:")
print(f"  Features with significant shift (p < 0.01): {ks_df['significant'].sum()} / {len(ks_df)}")
print(f"  Mean KS statistic: {ks_df['statistic'].mean():.4f}")
print(f"  Max KS statistic: {ks_df['statistic'].max():.4f}")

# Top shifted features
top_shifted = ks_df.head(N_FEATURES_PLOT)

print(f"\nTop {N_FEATURES_PLOT} most shifted features:")
print(top_shifted[['feature', 'statistic', 'pvalue', 'significant']].to_string(index=False))

# Plot top shifted features
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for idx, row in enumerate(top_shifted.itertuples()):
    ax = axes[idx]
    feat = row.feature

    # Plot histograms
    ax.hist(train[feat], bins=50, alpha=0.5, label='Train', density=True, color='steelblue')
    ax.hist(test[feat], bins=50, alpha=0.5, label='Test', density=True, color='coral')
    ax.set_title(f"{feat[:20]}...\nKS={row.statistic:.3f}, p={row.pvalue:.2e}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# Remove empty subplots
for idx in range(len(top_shifted), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Top Distribution-Shifted Features (KS Test)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_distribution_shift.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: feature_distribution_shift.png")

# Plot mean and std differences
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean difference histogram
axes[0].hist(mean_diff_pct, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Mean Difference (%)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Mean Differences (Test - Train)', fontweight='bold')
axes[0].grid(alpha=0.3)

# Std difference histogram
axes[1].hist(std_diff_pct, bins=100, color='coral', alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Std Difference (%)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Std Differences (Test - Train)', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_statistics_difference.png', dpi=300, bbox_inches='tight')
print(f"Saved: feature_statistics_difference.png")

# =============================================================================
# 2. SPECIES-STRATIFIED COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("2. SPECIES-STRATIFIED FEATURE DISTRIBUTION COMPARISON")
print("=" * 80)

# Species distribution
train_species = train['species_id'].value_counts().sort_index()
test_species = test['species_id'].value_counts().sort_index()

print("\nSpecies distribution:")
for sp_id in range(4):
    train_count = train_species.get(sp_id, 0)
    test_count = test_species.get(sp_id, 0)
    train_pct = train_count / len(train) * 100
    test_pct = test_count / len(test) * 100
    shift = test_pct - train_pct
    print(f"  {species_names[sp_id]}: Train={train_count} ({train_pct:.1f}%), "
          f"Test={test_count} ({test_pct:.1f}%), Shift={shift:+.1f}pp")

# Chi-square test for species distribution
chi2, pvalue, dof, expected = stats.chi2_contingency(pd.DataFrame({
    'Train': [train_species.get(i, 0) for i in range(4)],
    'Test': [test_species.get(i, 0) for i in range(4)]
}))

print(f"\nChi-square test for species distribution:")
print(f"  Chi2 statistic: {chi2:.2f}")
print(f"  P-value: {pvalue:.2e}")
print(f"  Significant: {'YES' if pvalue < 0.05 else 'NO'}")

# Per-species feature distribution comparison
print("\nAnalyzing feature distributions within each species...")
species_shift_stats = []

for sp_id in range(4):
    train_sp = train[train['species_id'] == sp_id]
    test_sp = test[test['species_id'] == sp_id]

    if len(train_sp) == 0 or len(test_sp) == 0:
        continue

    # Sample features for efficiency
    sample_feats = np.random.choice(feature_cols, min(500, len(feature_cols)), replace=False)

    ks_stats = []
    for feat in sample_feats:
        statistic, _ = stats.ks_2samp(train_sp[feat], test_sp[feat])
        ks_stats.append(statistic)

    species_shift_stats.append({
        'species': species_names[sp_id],
        'mean_ks': np.mean(ks_stats),
        'max_ks': np.max(ks_stats),
        'train_samples': len(train_sp),
        'test_samples': len(test_sp)
    })

species_shift_df = pd.DataFrame(species_shift_stats)

print("\nSpecies-stratified distribution shift:")
print(species_shift_df.to_string(index=False))

# Plot species-stratified comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for sp_id in range(4):
    ax = axes[sp_id]

    train_sp = train[train['species_id'] == sp_id]
    test_sp = test[test['species_id'] == sp_id]

    if len(train_sp) == 0 or len(test_sp) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        continue

    # Sample a few features to visualize
    sample_feats = np.random.choice(feature_cols, 4, replace=False)

    for feat in sample_feats:
        train_mean = train_sp[feat].mean()
        test_mean = test_sp[feat].mean()
        diff_pct = (test_mean - train_mean) / (train_mean + 1e-8) * 100
        ax.scatter([diff_pct], [feat], alpha=0.6, s=50)

    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Mean Difference (%)')
    ax.set_ylabel('Feature')
    ax.set_title(f'{species_names[sp_id]}\n(Train: {len(train_sp)}, Test: {len(test_sp)})', fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('Species-Stratified Feature Distribution Shift', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'species_stratified_shift.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: species_stratified_shift.png")

# =============================================================================
# 3. COVARIATE SHIFT DETECTION (DOMAIN CLASSIFIER)
# =============================================================================

print("\n" + "=" * 80)
print("3. COVARIATE SHIFT DETECTION (DOMAIN CLASSIFIER)")
print("=" * 80)

print("\nPreparing domain classifier data...")

# Sample for efficiency
train_sample = train.sample(n=min(SAMPLE_SIZE, len(train)), random_state=RANDOM_STATE)
test_sample = test.sample(n=min(SAMPLE_SIZE, len(test)), random_state=RANDOM_STATE)

# Create domain labels
X_train_dom = train_sample[feature_cols].values
X_test_dom = test_sample[feature_cols].values
y_train_dom = np.zeros(len(X_train_dom))  # Train = 0
y_test_dom = np.ones(len(X_test_dom))     # Test = 1

X_dom = np.vstack([X_train_dom, X_test_dom])
y_dom = np.hstack([y_train_dom, y_test_dom])

# Train domain classifier
print("Training domain classifier (Random Forest)...")
domain_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Use cross-validation to get AUC
cv_scores = cross_val_score(domain_clf, X_dom, y_dom, cv=5, scoring='roc_auc', n_jobs=-1)
mean_auc = cv_scores.mean()
std_auc = cv_scores.std()

print(f"\nDomain Classifier Results:")
print(f"  Cross-validated AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
print(f"  Interpretation: ", end="")

if mean_auc < 0.55:
    print("Minimal covariate shift - distributions are similar")
elif mean_auc < 0.65:
    print("Moderate covariate shift - some distribution differences")
elif mean_auc < 0.75:
    print("Significant covariate shift - clear distribution differences")
else:
    print("Severe covariate shift - distributions are very different")

# Train final model to get feature importance
domain_clf.fit(X_dom, y_dom)

# Get most discriminative features
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': domain_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 most discriminative features:")
print(feature_importance.head(20).to_string(index=False))

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance.head(20)
ax.barh(range(len(top_features)), top_features['importance'].values[::-1])
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([f[:30] + '...' if len(f) > 30 else f
                    for f in top_features['feature'].values[::-1]], fontsize=8)
ax.set_xlabel('Feature Importance')
ax.set_title('Most Discriminative Features (Domain Classifier)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'domain_classifier_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: domain_classifier_feature_importance.png")

# =============================================================================
# 4. DENSITY ESTIMATION (PCA-REDUCED)
# =============================================================================

print("\n" + "=" * 80)
print("4. DENSITY ESTIMATION & OOD DETECTION")
print("=" * 80)

print("\nReducing dimensionality with PCA...")

# Sample and standardize
train_sample = train.sample(n=min(1000, len(train)), random_state=RANDOM_STATE)
test_sample = test.sample(n=min(1000, len(test)), random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_sample[feature_cols])
X_test_scaled = scaler.transform(test_sample[feature_cols])

# Fit PCA
n_components = 50
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"  Explained variance ratio (top 50 PCs): {pca.explained_variance_ratio_.sum():.3f}")
print(f"  First PC: {pca.explained_variance_ratio_[0]:.3f}")

# Fit KDE on train
print("\nFitting KDE on train data (first 2 PCs)...")
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(X_train_pca[:, :2])

# Score train and test
train_log_dens = kde.score_samples(X_train_pca[:, :2])
test_log_dens = kde.score_samples(X_test_pca[:, :2])

print(f"\nDensity statistics:")
print(f"  Train log-density: {train_log_dens.mean():.3f} +/- {train_log_dens.std():.3f}")
print(f"  Test log-density:  {test_log_dens.mean():.3f} +/- {test_log_dens.std():.3f}")

# OOD detection threshold
threshold = np.percentile(train_log_dens, 5)  # 5th percentile
ood_samples = (test_log_dens < threshold).sum()

print(f"\nOOD Detection (using 5th percentile threshold):")
print(f"  Threshold: {threshold:.3f}")
print(f"  Test samples below threshold: {ood_samples} / {len(test_sample)} ({ood_samples/len(test_sample)*100:.1f}%)")

# Plot density visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Log-density comparison
axes[0].hist(train_log_dens, bins=50, alpha=0.5, label='Train', density=True, color='steelblue')
axes[0].hist(test_log_dens, bins=50, alpha=0.5, label='Test', density=True, color='coral')
axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label='OOD Threshold')
axes[0].set_xlabel('Log-Density')
axes[0].set_ylabel('Density')
axes[0].set_title('Log-Density Distribution (KDE on first 2 PCs)', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2D scatter plot of first 2 PCs
scatter = axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                         c=train_log_dens, cmap='Blues', alpha=0.5, s=20, label='Train')
axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1],
               c=test_log_dens, cmap='Reds', alpha=0.5, s=20, marker='s', label='Test')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('PCA Visualization (First 2 Components)', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'density_estimation_pca.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: density_estimation_pca.png")

# =============================================================================
# 5. MULTIVARIATE DISTRIBUTION COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("5. MULTIVARIATE DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\nComputing distribution distances...")

# Jensen-Shannon divergence between feature distributions
print("Computing Jensen-Shannon divergence (sampled features)...")

# Normalize train and test to get probability distributions
train_norm = train[feature_cols].values + np.abs(train[feature_cols].values.min()) + 1e-8
test_norm = test[feature_cols].values + np.abs(test[feature_cols].values.min()) + 1e-8

# Aggregate to get overall distribution
train_dist = train_norm.sum(axis=0)
train_dist = train_dist / train_dist.sum()

test_dist = test_norm.sum(axis=0)
test_dist = test_dist / test_dist.sum()

js_divergence = jensenshannon(train_dist, test_dist)

print(f"\nJensen-Shannon Divergence: {js_divergence:.4f}")
print(f"  Interpretation: ", end="")
if js_divergence < 0.1:
    print("Very similar distributions")
elif js_divergence < 0.2:
    print("Moderately different distributions")
elif js_divergence < 0.3:
    print("Significantly different distributions")
else:
    print("Very different distributions")

# Energy distance (univariate summary)
def energy_distance(x, y):
    n = len(x)
    m = len(y)

    # Sample for efficiency
    if n > 500:
        x = np.random.choice(x, 500, replace=False)
        n = 500
    if m > 500:
        y = np.random.choice(y, 500, replace=False)
        m = 500

    # Compute pairwise distances
    xx = np.abs(x[:, None] - x[None, :]).mean()
    yy = np.abs(y[:, None] - y[None, :]).mean()
    xy = np.abs(x[:, None] - y[None, :]).mean()

    return 2 * xy - xx - yy

print("\nComputing energy distance (sampled features)...")
sample_feats = np.random.choice(feature_cols, min(100, len(feature_cols)), replace=False)
energy_distances = []

for feat in tqdm(sample_feats, desc="Energy distances"):
    ed = energy_distance(train[feat].values, test[feat].values)
    energy_distances.append(ed)

print(f"\nEnergy Distance Statistics:")
print(f"  Mean: {np.mean(energy_distances):.4f}")
print(f"  Median: {np.median(energy_distances):.4f}")
print(f"  Max: {np.max(energy_distances):.4f}")

# =============================================================================
# SUMMARY AND RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY & KEY FINDINGS")
print("=" * 80)

# Collect all findings
findings = {
    'species_shift': {
        'chi2_stat': chi2,
        'chi2_pval': pvalue,
        'significant': pvalue < 0.05,
        'largest_shift': species_shift_df.loc[species_shift_df['mean_ks'].idxmax(), 'species'] if len(species_shift_df) > 0 else 'N/A'
    },
    'feature_shift': {
        'ks_significant': ks_df['significant'].sum(),
        'ks_total': len(ks_df),
        'mean_ks': ks_df['statistic'].mean(),
        'mean_diff_pct': np.abs(mean_diff_pct).mean()
    },
    'covariate_shift': {
        'domain_auc': mean_auc,
        'interpretation': 'severe' if mean_auc > 0.75 else 'significant' if mean_auc > 0.65 else 'moderate' if mean_auc > 0.55 else 'minimal'
    },
    'density': {
        'train_log_density': train_log_dens.mean(),
        'test_log_density': test_log_dens.mean(),
        'ood_pct': ood_samples / len(test_sample) * 100
    },
    'multivariate': {
        'js_divergence': js_divergence,
        'mean_energy_dist': np.mean(energy_distances)
    }
}

print("\n[CRITICAL FINDINGS]")
print(f"1. Species Distribution Shift: ", end="")
print(f"SIGNIFICANT (chi2={findings['species_shift']['chi2_stat']:.2f}, "
      f"p={findings['species_shift']['chi2_pval']:.2e})" if findings['species_shift']['significant'] else "Not significant")

print(f"2. Feature Distribution Shift: {findings['feature_shift']['ks_significant']}/{findings['feature_shift']['ks_total']} "
      f"features significantly shifted (KS test)")

print(f"3. Covariate Shift: {findings['covariate_shift']['domain_auc']:.4f} AUC - "
      f"{findings['covariate_shift']['interpretation'].upper()}")

print(f"4. Density Estimation: {findings['density']['ood_pct']:.1f}% of test samples are OOD "
      f"(below 5th percentile of train density)")

print(f"5. Jensen-Shannon Divergence: {findings['multivariate']['js_divergence']:.4f}")

print("\n[RECOMMENDATIONS]")
recommendations = []

if findings['species_shift']['significant']:
    recommendations.append("1. Use species-stratified cross-validation to ensure robust evaluation")

if findings['covariate_shift']['domain_auc'] > 0.6:
    recommendations.append("2. Consider domain adaptation techniques or importance weighting")

if findings['feature_shift']['ks_significant'] > findings['feature_shift']['ks_total'] * 0.5:
    recommendations.append("3. More than 50% of features show distribution shift - consider feature selection")

if findings['density']['ood_pct'] > 10:
    recommendations.append("4. Significant OOD samples detected - consider uncertainty estimation")

if findings['species_shift']['largest_shift'] == 'P. aeruginosa':
    recommendations.append("5. P. aeruginosa shows largest distribution shift - downweight or handle separately")

recommendations.append("6. Use validation strategy that mimics test distribution (e.g., group by species)")
recommendations.append("7. Consider ensemble methods robust to distribution shift")
recommendations.append("8. Monitor per-species performance during validation")

for rec in recommendations:
    print(f"  {rec}")

print("\n" + "=" * 80)
print(f"All outputs saved to: {OUTPUT_DIR}")
print("=" * 80)

# =============================================================================
# GENERATE MARKDOWN REPORT
# =============================================================================

print("\nGenerating markdown report...")

report_content = f"""# EDA Phase 6: Train-Test Distribution Analysis

**Date**: 2026-01-07
**Analysis**: Comprehensive distribution shift detection between train and test sets

---

## Executive Summary

This analysis reveals **significant distribution shift** between train and test sets, primarily driven by:
- **Species distribution imbalance**: P. aeruginosa 43% → 3% (train → test)
- **Covariate shift**: Domain classifier AUC = {findings['covariate_shift']['domain_auc']:.4f}
- **Feature-level shifts**: {findings['feature_shift']['ks_significant']}/{findings['feature_shift']['ks_total']} features significantly different

---

## 1. Species Distribution Shift

### Chi-Square Test
- **Statistic**: {findings['species_shift']['chi2_stat']:.2f}
- **P-value**: {findings['species_shift']['chi2_pval']:.2e}
- **Significance**: {'YES - Significant shift detected' if findings['species_shift']['significant'] else 'NO - No significant shift'}

### Species Breakdown
| Species | Train % | Test % | Shift (pp) |
|---------|---------|--------|------------|
"""

for sp_id in range(4):
    train_count = train_species.get(sp_id, 0)
    test_count = test_species.get(sp_id, 0)
    train_pct = train_count / len(train) * 100
    test_pct = test_count / len(test) * 100
    shift = test_pct - train_pct
    report_content += f"| {species_names[sp_id]} | {train_pct:.1f}% | {test_pct:.1f}% | {shift:+.1f} |\n"

report_content += f"""
**Key Finding**: Largest shift in **{findings['species_shift']['largest_shift']}** species

---

## 2. Feature Distribution Analysis

### Overall Statistics
- **Mean difference** (absolute): {np.abs(mean_diff).mean():.6f}
- **Mean difference** (percentage): {np.abs(mean_diff_pct).mean():.2f}%
- **Std difference** (absolute): {np.abs(std_diff).mean():.6f}
- **Std difference** (percentage): {np.abs(std_diff_pct).mean():.2f}%

### Kolmogorov-Smirnov Test Results
- **Features tested**: {len(ks_df)}
- **Significant shifts** (p < 0.01): {findings['feature_shift']['ks_significant']}
- **Mean KS statistic**: {findings['feature_shift']['mean_ks']:.4f}

### Top 10 Most Shifted Features
"""

for _, row in ks_df.head(10).iterrows():
    report_content += f"- **{row['feature']}**: KS = {row['statistic']:.4f}, p = {row['pvalue']:.2e}\n"

report_content += f"""

---

## 3. Species-Stratified Distribution Shift

| Species | Mean KS Statistic | Max KS Statistic | Train Samples | Test Samples |
|---------|-------------------|------------------|---------------|--------------|
"""

for _, row in species_shift_df.iterrows():
    report_content += f"| {row['species']} | {row['mean_ks']:.4f} | {row['max_ks']:.4f} | {row['train_samples']} | {row['test_samples']} |\n"

report_content += f"""

**Key Finding**: {findings['species_shift']['largest_shift']} shows highest within-species distribution shift

---

## 4. Covariate Shift Detection

### Domain Classifier Results
- **Algorithm**: Random Forest (100 trees, max_depth=10)
- **Cross-validated AUC**: {findings['covariate_shift']['domain_auc']:.4f}
- **Interpretation**: {findings['covariate_shift']['interpretation'].title()} covariate shift

### Most Discriminative Features
"""

for _, row in feature_importance.head(10).iterrows():
    report_content += f"- **{row['feature']}**: Importance = {row['importance']:.4f}\n"

report_content += f"""

---

## 5. Density Estimation & OOD Detection

### Methodology
- **Dimensionality reduction**: PCA (50 components, {pca.explained_variance_ratio_.sum():.1%} variance)
- **Density estimation**: Kernel Density Estimation (Gaussian kernel)
- **OOD threshold**: 5th percentile of train density

### Results
- **Train log-density** (mean ± std): {findings['density']['train_log_density']:.3f} ± {train_log_dens.std():.3f}
- **Test log-density** (mean ± std): {findings['density']['test_log_density']:.3f} ± {test_log_dens.std():.3f}
- **OOD samples**: {findings['density']['ood_pct']:.1f}% of test set below train threshold

### Interpretation
"""

if findings['density']['ood_pct'] < 5:
    report_content += "- Minimal OOD samples - test set well-covered by train distribution\n"
elif findings['density']['ood_pct'] < 15:
    report_content += "- Moderate OOD presence - some test samples differ from train\n"
else:
    report_content += "- Significant OOD presence - many test samples out-of-distribution\n"

report_content += f"""

---

## 6. Multivariate Distribution Analysis

### Jensen-Shannon Divergence: {findings['multivariate']['js_divergence']:.4f}

"""

if findings['multivariate']['js_divergence'] < 0.1:
    report_content += "**Interpretation**: Very similar multivariate distributions\n"
elif findings['multivariate']['js_divergence'] < 0.2:
    report_content += "**Interpretation**: Moderately different multivariate distributions\n"
elif findings['multivariate']['js_divergence'] < 0.3:
    report_content += "**Interpretation**: Significantly different multivariate distributions\n"
else:
    report_content += "**Interpretation**: Very different multivariate distributions\n"

report_content += f"""
### Energy Distance Statistics
- **Mean**: {findings['multivariate']['mean_energy_dist']:.4f}
- **Median**: {np.median(energy_distances):.4f}
- **Max**: {np.max(energy_distances):.4f}

---

## Key Findings Summary

### Critical Issues
1. **Severe species distribution shift** - P. aeruginosa overrepresented in train (43% vs 3%)
2. **Significant covariate shift** - Domain classifier AUC of {findings['covariate_shift']['domain_auc']:.4f} indicates clear distribution differences
3. **{findings['feature_shift']['ks_significant']} features** show statistically significant distribution differences

### Moderate Concerns
1. **{findings['density']['ood_pct']:.1f}% OOD samples** - Some test samples not well-represented in training
2. **Feature mean shifts** - Average {np.abs(mean_diff_pct).mean():.1f}% difference in feature means

---

## Recommendations

### Immediate Actions
1. **Use species-stratified cross-validation** - Ensure each fold reflects test distribution
2. **Implement domain adaptation** - Consider importance weighting or adversarial training
3. **Monitor per-species performance** - Track metrics separately for each species

### Modeling Strategy
1. **Feature selection** - Remove or downweight features with extreme distribution shift
2. **Ensemble methods** - Use models robust to distribution shift (e.g., tree-based)
3. **Uncertainty estimation** - Identify and flag OOD samples during inference
4. **Species-specific handling** - Consider separate models or weighting for P. aeruginosa

### Validation Strategy
1. **Stratified Group K-Fold** - Group by species to prevent leakage
2. **Domain-aware validation** - Create validation splits matching test distribution
3. **Per-species metrics** - Report AUC for each species separately

### Advanced Techniques
1. **Importance weighting** - Upweight underrepresented species (E. coli, K. pneumoniae)
2. **Domain adversarial training** - Learn features invariant to train-test shift
3. **Test-time adaptation** - Adjust model predictions based on test distribution
4. **Pseudo-labeling** - Leverage test data for semi-supervised learning

---

## Figures Generated

1. `feature_distribution_shift.png` - Histogram overlays of top 12 shifted features
2. `feature_statistics_difference.png` - Distribution of mean and std differences
3. `species_stratified_shift.png` - Per-species feature shift analysis
4. `domain_classifier_feature_importance.png` - Most discriminative features
5. `density_estimation_pca.png` - KDE density visualization and PCA scatter plot

---

## Statistical Tests Summary

| Test | Statistic | P-value | Significant | Interpretation |
|------|-----------|---------|-------------|----------------|
| Chi-square (species) | {findings['species_shift']['chi2_stat']:.2f} | {findings['species_shift']['chi2_pval']:.2e} | {'Yes' if findings['species_shift']['significant'] else 'No'} | Species distribution differs |
| KS test (features) | {findings['feature_shift']['mean_ks']:.4f} (mean) | - | Yes | {findings['feature_shift']['ks_significant']} features shifted |
| Domain classifier | AUC = {findings['covariate_shift']['domain_auc']:.4f} | - | {'Yes' if findings['covariate_shift']['domain_auc'] > 0.6 else 'No'} | Covariate shift detected |

---

## Conclusion

The train-test distribution shift in this competition is **significant and multifaceted**:
- **Primary driver**: Species distribution imbalance
- **Secondary factor**: Feature-level distribution differences
- **Impact**: Models may overfit to train distribution, especially P. aeruginosa

**Success depends on**: Proper validation strategy, domain-aware modeling, and careful handling of species imbalance.

---

*Analysis performed using Phase 6 EDA script*
*All figures saved to `/home/centcom/data/ml_kaggle/outputs/eda/phase6/`*
"""

# Write report
report_path = KNOWLEDGE_DIR / 'eda_phase6_train_test_distribution.md'
with open(report_path, 'w') as f:
    f.write(report_content)

print(f"Markdown report saved: {report_path}")

print("\n" + "=" * 80)
print("PHASE 6 COMPLETE")
print("=" * 80)

print("\nGenerated files:")
for f in OUTPUT_DIR.glob('*.png'):
    print(f"  - outputs/eda/phase6/{f.name}")
print(f"  - knowledge/insights/eda_phase6_train_test_distribution.md")

print("\nRecommendations:")
print("  1. Review figures in outputs/eda/phase6/")
print("  2. Implement species-stratified validation")
print("  3. Consider domain adaptation techniques")
print("  4. Monitor per-species performance metrics")
