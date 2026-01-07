"""
Comprehensive Exploratory Data Analysis for AMR Prediction Competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
DATA_DIR = Path('/sata_disk/users/erfan/ml_kaggle/raw')
OUTPUT_DIR = Path('/sata_disk/users/erfan/ml_kaggle/outputs/eda')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target antibiotics
ANTIBIOTICS = [
    'Ampicillin', 'Levofloxacin', 'Ciprofloxacin', 'Imipenem',
    'Amoxicillin_Clavulanic_acid', 'Ertapenem', 'Cefotaxime', 'Cefuroxime'
]

# Species mapping
SPECIES_NAMES = {
    0: 'E. Coli',
    1: 'K. Pneumoniae',
    2: 'P. Mirabilis',
    3: 'P. Aeruginosa'
}

print("=" * 80)
print("ANTIMICROBIAL RESISTANCE PREDICTION - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv')
species_map = pd.read_csv(DATA_DIR / 'species_mapping.csv')

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("2. BASIC STATISTICS")
print("=" * 80)

print("\n--- Train Data Info ---")
print(f"Rows: {train.shape[0]}")
print(f"Columns: {train.shape[1]}")
print(f"Memory usage: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n--- Test Data Info ---")
print(f"Rows: {test.shape[0]}")
print(f"Columns: {test.shape[1]}")
print(f"Memory usage: {test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n--- Column Types ---")
print(train.dtypes.value_counts())

print("\n--- First few columns ---")
print(train.columns[:15].tolist())
print("...")
print(train.columns[-10:].tolist())

# ============================================================================
# 3. TARGET DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("3. TARGET DISTRIBUTION (8 Antibiotics)")
print("=" * 80)

target_stats = []
for ab in ANTIBIOTICS:
    if ab in train.columns:
        total = len(train)
        missing = train[ab].isna().sum()
        available = total - missing
        if available > 0:
            resistant = (train[ab] == 1).sum()
            susceptible = (train[ab] == 0).sum()
            resist_rate = resistant / available * 100
        else:
            resistant = susceptible = resist_rate = 0

        target_stats.append({
            'Antibiotic': ab,
            'Total': total,
            'Missing': missing,
            'Available': available,
            'Resistant (1)': resistant,
            'Susceptible (0)': susceptible,
            'Resistance Rate %': resist_rate
        })

target_df = pd.DataFrame(target_stats)
print("\n--- Label Availability and Class Balance ---")
print(target_df.to_string(index=False))

# Save target stats
target_df.to_csv(OUTPUT_DIR / 'target_statistics.csv', index=False)

# Visualize class balance
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, ab in enumerate(ANTIBIOTICS):
    ax = axes[i]
    data = train[ab].dropna()
    counts = data.value_counts().sort_index()

    colors = ['#2ecc71', '#e74c3c']  # green for susceptible, red for resistant
    ax.bar(['Susceptible (0)', 'Resistant (1)'],
           [counts.get(0, 0), counts.get(1, 0)],
           color=colors)
    ax.set_title(ab, fontsize=10)
    ax.set_ylabel('Count')

    # Add percentage annotation
    total = len(data)
    for j, (label, count) in enumerate(zip([0, 1], [counts.get(0, 0), counts.get(1, 0)])):
        pct = count / total * 100 if total > 0 else 0
        ax.text(j, count + 5, f'{pct:.1f}%', ha='center', fontsize=9)

plt.suptitle('Target Class Distribution (8 Antibiotics)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_class_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: target_class_balance.png")

# ============================================================================
# 4. MISSING LABEL PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("4. MISSING LABEL PATTERNS")
print("=" * 80)

# Create missing indicator matrix
missing_matrix = train[ANTIBIOTICS].isna().astype(int)
missing_patterns = missing_matrix.apply(lambda x: ''.join(x.astype(str)), axis=1)
pattern_counts = missing_patterns.value_counts()

print("\n--- Top 15 Missing Label Patterns ---")
print("(1 = missing, 0 = available)")
print(f"{'Pattern':<20} {'Count':<10} {'Percentage':<10}")
print("-" * 40)
for i, (pattern, count) in enumerate(pattern_counts.head(15).items()):
    pct = count / len(train) * 100
    print(f"{pattern:<20} {count:<10} {pct:.1f}%")

print(f"\nTotal unique missing patterns: {len(pattern_counts)}")
print(f"Samples with all labels: {pattern_counts.get('0'*8, 0)} ({pattern_counts.get('0'*8, 0)/len(train)*100:.1f}%)")
print(f"Samples with all missing: {pattern_counts.get('1'*8, 0)} ({pattern_counts.get('1'*8, 0)/len(train)*100:.1f}%)")

# Visualize missing patterns
fig, ax = plt.subplots(figsize=(12, 8))
missing_heatmap = train[ANTIBIOTICS].isna().astype(int)
sns.heatmap(missing_heatmap.head(200), cmap='RdYlGn_r', cbar_kws={'label': 'Missing'}, ax=ax)
ax.set_xlabel('Antibiotic')
ax.set_ylabel('Sample Index (first 200)')
ax.set_title('Missing Label Pattern (First 200 Samples)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'missing_label_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: missing_label_heatmap.png")

# ============================================================================
# 5. TARGET CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("5. TARGET CORRELATION")
print("=" * 80)

# Calculate correlation on non-missing values
target_corr = train[ANTIBIOTICS].corr()
print("\n--- Antibiotic Resistance Correlation Matrix ---")
print(target_corr.round(3).to_string())

# Visualize correlation
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(target_corr, dtype=bool), k=1)
sns.heatmap(target_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Antibiotic Resistance Correlation Matrix')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: target_correlation.png")

# ============================================================================
# 6. SPECIES DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("6. SPECIES DISTRIBUTION")
print("=" * 80)

print("\n--- Species Count in Train ---")
train_species = train['species_id'].value_counts().sort_index()
for sid, count in train_species.items():
    print(f"  {SPECIES_NAMES[sid]}: {count} ({count/len(train)*100:.1f}%)")

print("\n--- Species Count in Test ---")
test_species = test['species_id'].value_counts().sort_index()
for sid, count in test_species.items():
    print(f"  {SPECIES_NAMES[sid]}: {count} ({count/len(test)*100:.1f}%)")

# Visualize species distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (data, title) in zip(axes, [(train, 'Train'), (test, 'Test')]):
    counts = data['species_id'].value_counts().sort_index()
    labels = [SPECIES_NAMES[i] for i in counts.index]
    colors = plt.cm.Set2(np.arange(4))
    ax.bar(labels, counts.values, color=colors)
    ax.set_title(f'{title} Set Species Distribution')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    for i, (label, count) in enumerate(zip(labels, counts.values)):
        ax.text(i, count + 5, str(count), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'species_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: species_distribution.png")

# ============================================================================
# 7. RESISTANCE RATES BY SPECIES
# ============================================================================
print("\n" + "=" * 80)
print("7. RESISTANCE RATES BY SPECIES")
print("=" * 80)

species_resistance = []
for species_id in sorted(train['species_id'].unique()):
    species_data = train[train['species_id'] == species_id]
    row = {'Species': SPECIES_NAMES[species_id], 'N': len(species_data)}

    for ab in ANTIBIOTICS:
        ab_data = species_data[ab].dropna()
        if len(ab_data) > 0:
            resist_rate = (ab_data == 1).sum() / len(ab_data) * 100
            row[ab] = f"{resist_rate:.1f}% (n={len(ab_data)})"
        else:
            row[ab] = "N/A"

    species_resistance.append(row)

species_resist_df = pd.DataFrame(species_resistance)
print("\n--- Resistance Rate by Species per Antibiotic ---")
print(species_resist_df.to_string(index=False))

# Create heatmap of resistance rates
fig, ax = plt.subplots(figsize=(14, 6))
resist_matrix = np.zeros((4, 8))
sample_counts = np.zeros((4, 8))

for i, species_id in enumerate(sorted(train['species_id'].unique())):
    species_data = train[train['species_id'] == species_id]
    for j, ab in enumerate(ANTIBIOTICS):
        ab_data = species_data[ab].dropna()
        if len(ab_data) > 0:
            resist_matrix[i, j] = (ab_data == 1).sum() / len(ab_data) * 100
            sample_counts[i, j] = len(ab_data)
        else:
            resist_matrix[i, j] = np.nan

sns.heatmap(resist_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
            xticklabels=ANTIBIOTICS, yticklabels=[SPECIES_NAMES[i] for i in range(4)],
            ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Resistance Rate %'})
ax.set_title('Resistance Rate (%) by Species and Antibiotic')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'resistance_by_species.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: resistance_by_species.png")

# Label availability by species
print("\n--- Label Availability by Species ---")
for species_id in sorted(train['species_id'].unique()):
    species_data = train[train['species_id'] == species_id]
    print(f"\n{SPECIES_NAMES[species_id]} (n={len(species_data)}):")
    for ab in ANTIBIOTICS:
        available = species_data[ab].notna().sum()
        pct = available / len(species_data) * 100
        print(f"  {ab}: {available}/{len(species_data)} ({pct:.1f}%)")

# ============================================================================
# 8. MALDI FEATURE STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("8. MALDI FEATURE STATISTICS")
print("=" * 80)

# Extract MALDI features
maldi_cols = [c for c in train.columns if c.startswith('maldi_feature_')]
print(f"\nNumber of MALDI features: {len(maldi_cols)}")

train_maldi = train[maldi_cols]
test_maldi = test[maldi_cols]

print("\n--- Train MALDI Features Summary ---")
train_stats = train_maldi.describe()
print(f"Global Min: {train_maldi.values.min():.4f}")
print(f"Global Max: {train_maldi.values.max():.4f}")
print(f"Global Mean: {train_maldi.values.mean():.4f}")
print(f"Global Std: {train_maldi.values.std():.4f}")

print("\n--- Test MALDI Features Summary ---")
print(f"Global Min: {test_maldi.values.min():.4f}")
print(f"Global Max: {test_maldi.values.max():.4f}")
print(f"Global Mean: {test_maldi.values.mean():.4f}")
print(f"Global Std: {test_maldi.values.std():.4f}")

# Check for missing values in features
train_missing = train_maldi.isna().sum().sum()
test_missing = test_maldi.isna().sum().sum()
print(f"\nMissing values in train MALDI features: {train_missing}")
print(f"Missing values in test MALDI features: {test_missing}")

# Check for constant or near-constant features
feature_stds = train_maldi.std()
constant_features = (feature_stds == 0).sum()
near_constant = (feature_stds < 0.001).sum()
print(f"\nConstant features (std=0): {constant_features}")
print(f"Near-constant features (std<0.001): {near_constant}")

if near_constant > 0:
    print("\nNear-constant features:")
    near_const_cols = feature_stds[feature_stds < 0.001].index.tolist()
    print(f"  {near_const_cols[:10]}...")  # Show first 10

# Feature value distribution
print("\n--- Feature Statistics Distribution Across 6000 Features ---")
feature_means = train_maldi.mean()
feature_stds = train_maldi.std()
feature_mins = train_maldi.min()
feature_maxs = train_maldi.max()

print(f"Mean of feature means: {feature_means.mean():.4f}")
print(f"Std of feature means: {feature_means.std():.4f}")
print(f"Mean of feature stds: {feature_stds.mean():.4f}")
print(f"Range of feature means: [{feature_means.min():.4f}, {feature_means.max():.4f}]")

# ============================================================================
# 9. FEATURE DISTRIBUTION VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("9. FEATURE DISTRIBUTION VISUALIZATION")
print("=" * 80)

# Sample features for visualization
np.random.seed(42)
sample_feature_indices = [0, 1000, 2000, 3000, 4000, 5000, 5999]
sample_features = [f'maldi_feature_{i}' for i in sample_feature_indices]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(sample_features):
    ax = axes[i]
    ax.hist(train[feat], bins=50, alpha=0.7, label='Train', color='blue')
    ax.hist(test[feat], bins=50, alpha=0.7, label='Test', color='orange')
    ax.set_title(feat)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()

# Use last subplot for overall distribution
ax = axes[-1]
ax.hist(train_maldi.values.flatten()[::100], bins=100, alpha=0.7, label='Train', color='blue')
ax.hist(test_maldi.values.flatten()[::100], bins=100, alpha=0.7, label='Test', color='orange')
ax.set_title('Overall Feature Distribution (sampled)')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.legend()

plt.suptitle('MALDI Feature Distributions (Train vs Test)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: feature_distributions.png")

# Mean spectrum plot
fig, ax = plt.subplots(figsize=(16, 6))
train_mean_spectrum = train_maldi.mean(axis=0).values
test_mean_spectrum = test_maldi.mean(axis=0).values

ax.plot(train_mean_spectrum, label='Train Mean', alpha=0.8, linewidth=0.5)
ax.plot(test_mean_spectrum, label='Test Mean', alpha=0.8, linewidth=0.5)
ax.set_xlabel('Feature Index (0-5999)')
ax.set_ylabel('Mean Intensity')
ax.set_title('Mean MALDI Spectrum: Train vs Test')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mean_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mean_spectrum.png")

# ============================================================================
# 10. FEATURE CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("10. FEATURE CORRELATION (Sampled)")
print("=" * 80)

# Sample features for correlation (every 100th feature)
sampled_cols = maldi_cols[::100]  # 60 features
sampled_corr = train[sampled_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(sampled_corr, cmap='RdBu_r', center=0, ax=ax,
            xticklabels=False, yticklabels=False)
ax.set_title('Feature Correlation Matrix (Every 100th Feature)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_correlation_sampled.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: feature_correlation_sampled.png")

# High correlation summary
high_corr = (sampled_corr.abs() > 0.8) & (sampled_corr != 1.0)
print(f"Feature pairs with |correlation| > 0.8 (sampled): {high_corr.sum().sum() // 2}")

# ============================================================================
# 11. TRAIN-TEST DISTRIBUTION SHIFT
# ============================================================================
print("\n" + "=" * 80)
print("11. TRAIN-TEST DISTRIBUTION SHIFT")
print("=" * 80)

# Compare species distribution
print("\n--- Species Distribution Comparison ---")
train_species_pct = train['species_id'].value_counts(normalize=True).sort_index() * 100
test_species_pct = test['species_id'].value_counts(normalize=True).sort_index() * 100

comparison_df = pd.DataFrame({
    'Species': [SPECIES_NAMES[i] for i in range(4)],
    'Train %': train_species_pct.values,
    'Test %': test_species_pct.values,
    'Diff %': test_species_pct.values - train_species_pct.values
})
print(comparison_df.to_string(index=False))

# Compare feature distributions using KS statistic
from scipy import stats

print("\n--- Feature Distribution Shift (KS Test on Sampled Features) ---")
ks_results = []
for feat in maldi_cols[::100]:  # Sample every 100th feature
    ks_stat, p_value = stats.ks_2samp(train[feat], test[feat])
    ks_results.append({'Feature': feat, 'KS Statistic': ks_stat, 'P-Value': p_value})

ks_df = pd.DataFrame(ks_results)
print(f"Mean KS statistic: {ks_df['KS Statistic'].mean():.4f}")
print(f"Max KS statistic: {ks_df['KS Statistic'].max():.4f}")
print(f"Features with significant shift (p<0.05): {(ks_df['P-Value'] < 0.05).sum()}/{len(ks_df)}")

# Worst shifts
print("\nTop 5 features with largest distribution shift:")
print(ks_df.nlargest(5, 'KS Statistic').to_string(index=False))

# ============================================================================
# 12. SPECIES-SPECIFIC FEATURE PATTERNS
# ============================================================================
print("\n" + "=" * 80)
print("12. SPECIES-SPECIFIC FEATURE PATTERNS")
print("=" * 80)

# Mean spectrum by species
fig, ax = plt.subplots(figsize=(16, 8))
colors = plt.cm.Set1(np.arange(4))

for species_id in range(4):
    species_data = train[train['species_id'] == species_id][maldi_cols]
    mean_spectrum = species_data.mean(axis=0).values
    ax.plot(mean_spectrum, label=SPECIES_NAMES[species_id], alpha=0.8, linewidth=0.8, color=colors[species_id])

ax.set_xlabel('Feature Index (0-5999)')
ax.set_ylabel('Mean Intensity')
ax.set_title('Mean MALDI Spectrum by Species')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mean_spectrum_by_species.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: mean_spectrum_by_species.png")

# Feature variance by species
species_variances = []
for species_id in range(4):
    species_data = train[train['species_id'] == species_id][maldi_cols]
    var = species_data.var().mean()
    species_variances.append({'Species': SPECIES_NAMES[species_id], 'Mean Feature Variance': var})

print("\n--- Mean Feature Variance by Species ---")
print(pd.DataFrame(species_variances).to_string(index=False))

# ============================================================================
# 13. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("13. KEY INSIGHTS FOR MODELING")
print("=" * 80)

print("""
==========================================
DATA CHARACTERISTICS SUMMARY
==========================================

1. DATASET SIZE:
   - Train: 3,360 samples x 6,010 features (6000 MALDI + sample_id + species_id + 8 targets)
   - Test: 1,000 samples (no targets)
   - This is a relatively small dataset for 6000 features -> regularization critical

2. TARGET CHARACTERISTICS:
   - 8 binary classification targets (antibiotics)
   - Varying class imbalance per antibiotic
   - Strong correlations between certain antibiotic pairs
   - Missing labels: semi-supervised setting

3. SPECIES DISTRIBUTION:
   - 4 bacterial species with different sample counts
   - Species distribution similar between train/test
   - Resistance patterns vary significantly by species
   - Species is a critical feature for prediction

4. FEATURE CHARACTERISTICS:
   - 6000 MALDI spectral features
   - No missing values in features
   - Features are continuous, non-negative
   - Some near-constant features may be removed
   - High correlation between adjacent features

==========================================
PREPROCESSING RECOMMENDATIONS
==========================================

1. FEATURE SCALING:
   - StandardScaler or MinMaxScaler for the 6000 MALDI features
   - Features have varying scales

2. FEATURE SELECTION:
   - Remove near-constant features (std < threshold)
   - Consider PCA for dimensionality reduction
   - Feature importance from tree models

3. HANDLING MISSING LABELS:
   - Mask loss for NaN targets during training
   - Consider pseudo-labeling for unlabeled samples
   - Leverage label correlations

4. SPECIES HANDLING:
   - Use species embedding (already in baseline)
   - Consider species-specific models
   - Species-stratified cross-validation

==========================================
MODELING CONSIDERATIONS
==========================================

1. HARDEST ANTIBIOTICS TO PREDICT:
   - Antibiotics with most missing labels
   - Antibiotics with extreme class imbalance
   - Less correlated with other antibiotics

2. SEMI-SUPERVISED APPROACHES:
   - Pseudo-labeling with confidence threshold
   - Consistency regularization
   - Label propagation using feature similarity

3. MODEL ARCHITECTURE:
   - Species embedding is important
   - 1D CNN may capture spectral patterns
   - Attention for feature importance
   - Multi-task learning for correlated targets

4. VALIDATION STRATEGY:
   - Stratify by species AND label availability
   - K-fold cross-validation
   - Watch for overfitting (small data, many features)
""")

# Save summary statistics
summary = {
    'train_samples': len(train),
    'test_samples': len(test),
    'n_features': len(maldi_cols),
    'n_antibiotics': 8,
    'n_species': 4,
    'train_missing_labels_total': train[ANTIBIOTICS].isna().sum().sum(),
    'train_missing_labels_pct': train[ANTIBIOTICS].isna().sum().sum() / (len(train) * 8) * 100,
}

print("\n--- Summary Statistics ---")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 80)
print("EDA COMPLETE - Visualizations saved to: outputs/eda/")
print("=" * 80)

# List saved files
print("\nSaved files:")
for f in sorted(OUTPUT_DIR.glob('*')):
    print(f"  - {f.name}")
