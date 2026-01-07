"""
Phase 1: Data Profiling & Quality Assessment
============================================

This script performs comprehensive data profiling for the AMR prediction competition,
including schema analysis, missing value detection, duplicate checking, and train/test alignment.

Outputs:
    - Figures saved to outputs/eda/phase1/
    - Console output with detailed statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set paths
PROJECT_ROOT = Path('/sata_disk/users/erfan/ml_kaggle')
RAW_DIR = PROJECT_ROOT / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eda' / 'phase1'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("PHASE 1: DATA PROFILING & QUALITY ASSESSMENT")
print("=" * 80)
print()

# =============================================================================
# 1.1 SCHEMA & TYPES
# =============================================================================

print("\n" + "=" * 80)
print("1.1 SCHEMA & TYPES")
print("=" * 80)

# Load data
train = pd.read_csv(RAW_DIR / 'train.csv')
test = pd.read_csv(RAW_DIR / 'test.csv')

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\n--- Column Types ---")
print("\nTrain dtypes:")
print(train.dtypes.value_counts())

print("\n--- Memory Usage ---")
train_memory = train.memory_usage(deep=True).sum() / 1024**2
test_memory = test.memory_usage(deep=True).sum() / 1024**2
print(f"\nTrain memory: {train_memory:.2f} MB")
print(f"Test memory: {test_memory:.2f} MB")
print(f"Total: {train_memory + test_memory:.2f} MB")

print("\n--- Column Names ---")
feature_cols = [col for col in train.columns if col.startswith('maldi_feature_')]
target_cols = ['Ampicillin', 'Levofloxacin', 'Ciprofloxacin', 'Imipenem',
               'Amoxicillin_Clavulanic_acid', 'Ertapenem', 'Cefotaxime', 'Cefuroxime']

print(f"\nIdentifiers: sample_id, species_id")
print(f"Feature columns: {len(feature_cols)} (maldi_feature_0 to maldi_feature_{len(feature_cols)-1})")
print(f"Target columns: {len(target_cols)}")
print(f"  {', '.join(target_cols)}")

print("\n--- Data Type Validation ---")
print("\nChecking for unexpected dtypes...")
unexpected_dtypes = []
for col in train.columns:
    if col in ['sample_id', 'species_id']:
        if train[col].dtype not in ['int64', 'int32']:
            unexpected_dtypes.append(f"{col}: {train[col].dtype}")
    elif col in target_cols:
        if train[col].dtype not in ['float64', 'float64']:
            unexpected_dtypes.append(f"{col}: {train[col].dtype}")
    elif col.startswith('maldi_feature_'):
        if train[col].dtype not in ['float64', 'float64']:
            unexpected_dtypes.append(f"{col}: {train[col].dtype}")

if unexpected_dtypes:
    print("Unexpected dtypes found:")
    for item in unexpected_dtypes:
        print(f"  - {item}")
else:
    print("All dtypes are as expected.")

# =============================================================================
# 1.2 MISSING VALUES ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("1.2 MISSING VALUES ANALYSIS")
print("=" * 80)

# Overall missingness
print("\n--- Missing Values Overview ---")
missing_counts = train[target_cols].isna().sum()
missing_pct = (missing_counts / len(train) * 100).round(2)

missing_df = pd.DataFrame({
    'Count': missing_counts,
    'Percentage': missing_pct,
    'Total': len(train)
})
missing_df['Labeled'] = missing_df['Total'] - missing_df['Count']
missing_df['Labeled_Pct'] = ((missing_df['Labeled'] / missing_df['Total']) * 100).round(2)

print("\nMissing values by target:")
print(missing_df[['Count', 'Percentage', 'Labeled', 'Labeled_Pct']])

# Check for missing in features and metadata
print("\n--- Missing in Features/Metadata ---")
feature_missing = train[feature_cols].isna().sum().sum()
metadata_missing = train[['sample_id', 'species_id']].isna().sum().sum()

print(f"\nMissing in MALDI features: {feature_missing}")
print(f"Missing in sample_id: {train['sample_id'].isna().sum()}")
print(f"Missing in species_id: {train['species_id'].isna().sum()}")

# Missingness by species
print("\n--- Missingness by Species ---")
species_names = {0: 'E. coli', 1: 'K. pneumoniae', 2: 'P. mirabilis', 3: 'P. aeruginosa'}
train['species_name'] = train['species_id'].map(species_names)

species_missing = pd.DataFrame()
for species_id, species_name in species_names.items():
    species_data = train[train['species_id'] == species_id]
    row = {
        'Species': species_name,
        'Samples': len(species_data),
    }
    for target in target_cols:
        missing_count = species_data[target].isna().sum()
        missing_pct = (missing_count / len(species_data) * 100) if len(species_data) > 0 else 0
        row[f'{target}_missing'] = missing_count
        row[f'{target}_pct'] = round(missing_pct, 1)
    species_missing = pd.concat([species_missing, pd.DataFrame([row])], ignore_index=True)

print("\nMissingness breakdown by species:")
print_cols = ['Species', 'Samples'] + [f'{t}_missing' for t in target_cols]
print(species_missing[print_cols].to_string(index=False))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap of missing values (sample subset for readability)
sample_size = min(500, len(train))
sample_indices = np.random.choice(len(train), sample_size, replace=False)
train_sample = train.iloc[sample_indices]

missing_matrix = train_sample[target_cols].isna().astype(int)
sns.heatmap(missing_matrix.T, cbar=False, cmap='Reds', ax=axes[0],
            xticklabels=False, yticklabels=True)
axes[0].set_title(f'Missing Values Heatmap (Random {sample_size} samples)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Samples')
axes[0].set_ylabel('Targets')

# Bar chart of missingness by species
species_missing_plot = species_missing[['Species'] + [f'{t}_pct' for t in target_cols]].set_index('Species')
species_missing_plot.columns = [t.replace('_', ' ') for t in target_cols]
species_missing_plot.T.plot(kind='bar', ax=axes[1], rot=45, fontsize=9)
axes[1].set_title('Missing Labels Percentage by Species', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Missing Percentage (%)')
axes[1].set_xlabel('Targets')
axes[1].legend(title='Species', fontsize=8)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'missing_values_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: missing_values_analysis.png")

# Missing value patterns
print("\n--- Missing Value Patterns ---")
# Count how many labels are missing per sample
train['n_missing_labels'] = train[target_cols].isna().sum(axis=1)
missing_label_counts = train['n_missing_labels'].value_counts().sort_index()

print("\nSamples by number of missing labels:")
for n_missing, count in missing_label_counts.items():
    pct = (count / len(train) * 100)
    print(f"  {n_missing} missing: {count} samples ({pct:.1f}%)")

# Clean up temporary column
train.drop('n_missing_labels', axis=1, inplace=True)

# =============================================================================
# 1.3 DUPLICATE DETECTION
# =============================================================================

print("\n" + "=" * 80)
print("1.3 DUPLICATE DETECTION")
print("=" * 80)

# Check duplicate sample_ids
print("\n--- Duplicate Sample IDs ---")
duplicate_sample_ids = train['sample_id'].duplicated().sum()
print(f"Duplicate sample_id in train: {duplicate_sample_ids}")

if duplicate_sample_ids > 0:
    dup_ids = train[train['sample_id'].duplicated(keep=False)]['sample_id'].unique()
    print(f"Number of unique duplicated IDs: {len(dup_ids)}")
    print(f"Example duplicated IDs: {dup_ids[:5]}")
else:
    print("All sample_id values are unique.")

# Check for duplicate feature rows (excluding sample_id)
print("\n--- Duplicate Feature Rows ---")
feature_cols_with_species = ['species_id'] + feature_cols
duplicate_features = train[feature_cols_with_species].duplicated().sum()
print(f"Duplicate feature rows (including species): {duplicate_features}")

# Check if same sample_id appears in test
print("\n--- Train/Test Overlap ---")
train_ids = set(train['sample_id'])
test_ids = set(test['sample_id'])
overlap = train_ids & test_ids
print(f"Sample IDs in both train and test: {len(overlap)}")

if len(overlap) > 0:
    print(f"WARNING: {len(overlap)} sample IDs appear in both train and test!")
    print(f"Example overlapping IDs: {list(overlap)[:5]}")

# Check exact duplicate rows across train and test
print("\n--- Cross-Dataset Duplicate Features ---")
print("Skipping full cross-dataset duplicate check (computationally expensive)")
print("Use hashing-based approach if needed for production analysis")

# =============================================================================
# 1.4 TRAIN/TEST ALIGNMENT
# =============================================================================

print("\n" + "=" * 80)
print("1.4 TRAIN/TEST ALIGNMENT")
print("=" * 80)

print("\n--- Feature Column Alignment ---")
train_features = set(train.columns)
test_features = set(test.columns)

target_only = train_features - test_features
test_only = test_features - train_features

print(f"Columns only in train: {len(target_only)}")
if target_only:
    print(f"  {sorted(target_only)}")

print(f"\nColumns only in test: {len(test_only)}")
if test_only:
    print(f"  {sorted(test_only)}")

# Check if feature columns match
train_feature_cols = set(feature_cols)
test_feature_cols = set([col for col in test.columns if col.startswith('maldi_feature_')])

if train_feature_cols == test_feature_cols:
    print("\nFeature columns are perfectly aligned.")
else:
    print("\nWARNING: Feature columns differ!")
    print(f"  Features only in train: {len(train_feature_cols - test_feature_cols)}")
    print(f"  Features only in test: {len(test_feature_cols - train_feature_cols)}")

# Basic statistics comparison
print("\n--- Feature Statistics Comparison ---")
print("Computing statistics (this may take a moment)...")

# Sample for efficiency
train_sample = train[feature_cols].sample(n=min(1000, len(train)), random_state=42)
test_sample = test[feature_cols].sample(n=min(1000, len(test)), random_state=42)

print("\nTrain feature statistics (from sample):")
print(f"  Mean: {train_sample.mean().mean():.6f}")
print(f"  Std:  {train_sample.std().mean():.6f}")
print(f"  Min:  {train_sample.min().min():.6f}")
print(f"  Max:  {train_sample.max().max():.6f}")

print("\nTest feature statistics (from sample):")
print(f"  Mean: {test_sample.mean().mean():.6f}")
print(f"  Std:  {test_sample.std().mean():.6f}")
print(f"  Min:  {test_sample.min().min():.6f}")
print(f"  Max:  {test_sample.max().max():.6f}")

# Species distribution comparison
print("\n--- Species Distribution Comparison ---")
train_species_dist = train['species_id'].value_counts().sort_index()
test_species_dist = test['species_id'].value_counts().sort_index()

species_comparison = pd.DataFrame({
    'Species': [species_names[i] for i in range(4)],
    'Train_Count': [train_species_dist.get(i, 0) for i in range(4)],
    'Train_Pct': [(train_species_dist.get(i, 0) / len(train) * 100) for i in range(4)],
    'Test_Count': [test_species_dist.get(i, 0) for i in range(4)],
    'Test_Pct': [(test_species_dist.get(i, 0) / len(test) * 100) for i in range(4)],
})
species_comparison['Shift_Pct'] = species_comparison['Test_Pct'] - species_comparison['Train_Pct']

print("\nSpecies distribution:")
print(species_comparison.to_string(index=False))

# Visualize species distribution shift
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of species distribution
x = np.arange(4)
width = 0.35

axes[0].bar(x - width/2, species_comparison['Train_Pct'], width, label='Train', color='steelblue')
axes[0].bar(x + width/2, species_comparison['Test_Pct'], width, label='Test', color='coral')
axes[0].set_xlabel('Species')
axes[0].set_ylabel('Percentage (%)')
axes[0].set_title('Species Distribution: Train vs Test', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(species_comparison['Species'], rotation=15, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Shift plot
axes[1].bar(species_comparison['Species'], species_comparison['Shift_Pct'],
            color=['green' if x > 0 else 'red' for x in species_comparison['Shift_Pct']])
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Species')
axes[1].set_ylabel('Percentage Point Shift')
axes[1].set_title('Species Distribution Shift (Test - Train)', fontweight='bold')
axes[1].set_xticklabels(species_comparison['Species'], rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'species_distribution_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: species_distribution_comparison.png")

# =============================================================================
# 1.5 FEATURE VALUE ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("1.5 FEATURE VALUE ANALYSIS")
print("=" * 80)

# Feature sparsity (use already computed samples)
print("\nComputing sparsity (using sample for efficiency)...")
train_sparse_pct = (train_sample == 0).sum().sum() / (len(train_sample) * len(feature_cols)) * 100
test_sparse_pct = (test_sample == 0).sum().sum() / (len(test_sample) * len(feature_cols)) * 100

print(f"\nFeature sparsity (zeros):")
print(f"  Train: {train_sparse_pct:.2f}%")
print(f"  Test:  {test_sparse_pct:.2f}%")

# Constant features (use sampling for efficiency)
print("\nIdentifying constant features (using sample for efficiency)...")
train_var_sample = train[feature_cols].sample(n=min(500, len(train)), random_state=42).var(axis=0)
constant_features_est = train_var_sample[train_var_sample == 0].index.tolist()
print(f"\nConstant features (zero variance, estimated from sample): {len(constant_features_est)}")

# Negative values (use already computed samples)
print("\nChecking for negative values (using sample for efficiency)...")
train_neg = (train_sample < 0).sum().sum()
test_neg = (test_sample < 0).sum().sum()
print(f"\nNegative values (estimated from sample):")
print(f"  Train: {train_neg}")
print(f"  Test:  {test_neg}")

# Feature value ranges (use already computed samples)
print(f"\nFeature value ranges (from sample):")
print(f"  Min:  {train_sample.min().min():.6f}")
print(f"  Max:  {train_sample.max().max():.6f}")
print(f"  Mean: {train_sample.mean().mean():.6f}")

# =============================================================================
# SUMMARY & KEY FINDINGS
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY & KEY FINDINGS")
print("=" * 80)

print("\n[DATA QUALITY ISSUES]")
issues = []

if missing_counts.sum() > 0:
    issues.append(f"- Missing labels: {missing_counts.sum()} total missing values across targets")
    max_missing = missing_counts.idxmax()
    max_missing_count = missing_counts[max_missing]
    max_missing_pct = (max_missing_count / len(train)) * 100
    issues.append(f"  - {max_missing} has highest missingness ({max_missing_count} samples, {max_missing_pct:.2f}%)")

if len(overlap) > 0:
    issues.append(f"- Sample ID overlap: {len(overlap)} IDs appear in both train and test")

if len(constant_features_est) > 0:
    issues.append(f"- Constant features: ~{len(constant_features_est)} features with zero variance (estimated)")

if duplicate_sample_ids > 0:
    issues.append(f"- Duplicate sample IDs: {duplicate_sample_ids} duplicates found")

if issues:
    for issue in issues:
        print(issue)
else:
    print("No critical data quality issues detected.")

print("\n[KEY INSIGHTS]")
print(f"1. Dataset size: {len(train)} train / {len(test)} test samples")
print(f"2. Features: {len(feature_cols)} MALDI-TOF binned intensities")
print(f"3. Targets: {len(target_cols)} antibiotic resistance labels")
print(f"4. Sparsity: {train_sparse_pct:.1f}% of feature values are zero")
print(f"5. Label completeness: {(1 - missing_counts.sum() / (len(train) * len(target_cols))) * 100:.1f}% of labels present")

print("\n[MODELING IMPLICATIONS]")
print("1. Semi-supervised learning: Many missing labels, especially Amoxicillin_Clavulanic_acid")
print(f"2. Feature selection: Consider removing ~{len(constant_features_est)} constant features")
print("3. Species shift: Significant distribution differences require careful validation")
print("4. Sparse data: Models should handle zero-inflated features (e.g., tree-based)")

print("\n" + "=" * 80)
print(f"All outputs saved to: {OUTPUT_DIR}")
print("=" * 80)

print("\nGenerated files:")
for f in OUTPUT_DIR.glob('*.png'):
    print(f"  - {f.name}")
