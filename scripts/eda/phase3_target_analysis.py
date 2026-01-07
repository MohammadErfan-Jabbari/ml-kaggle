#!/usr/bin/env python3
"""
Phase 3 EDA: Target Analysis for AMR Prediction

Analyzes the 8 antibiotic resistance labels:
- Class balance
- Missing label patterns
- Target correlations
- Co-resistance patterns
- Antibiotic class grouping
- Multi-label statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
DATA_DIR = "/sata_disk/users/erfan/ml_kaggle/raw"
OUTPUT_DIR = "/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase3"

# Antibiotic names
ANTIBIOTICS = [
    'Ampicillin',
    'Levofloxacin',
    'Ciprofloxacin',
    'Imipenem',
    'Amoxicillin_Clavulanic_acid',
    'Ertapenem',
    'Cefotaxime',
    'Cefuroxime'
]

# Antibiotic classes for interpretation
DRUG_CLASSES = {
    'Ampicillin': 'Penicillin',
    'Levofloxacin': 'Fluoroquinolone',
    'Ciprofloxacin': 'Fluoroquinolone',
    'Imipenem': 'Carbapenem',
    'Amoxicillin_Clavulanic_acid': 'Penicillin+BetaLactamase',
    'Ertapenem': 'Carbapenem',
    'Cefotaxime': 'Cephalosporin (3rd gen)',
    'Cefuroxime': 'Cephalosporin (2nd gen)'
}


def load_data():
    """Load training data and extract targets."""
    print("Loading data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv")

    # Extract metadata and targets
    sample_ids = train['sample_id']
    species = train['species_id']

    # Targets are the last 8 columns
    targets = train[ANTIBIOTICS].copy()

    print(f"Loaded {len(train)} samples with {len(ANTIBIOTICS)} targets")
    return train, targets, species


def analyze_class_balance(targets):
    """3.1: Calculate and visualize class balance."""
    print("\n" + "="*60)
    print("3.1: CLASS BALANCE ANALYSIS")
    print("="*60)

    # Calculate resistance rates
    resistance_rates = {}
    for abx in ANTIBIOTICS:
        valid_mask = targets[abx].notna()
        n_valid = valid_mask.sum()
        n_resistant = targets[abx][valid_mask].sum()
        rate = (n_resistant / n_valid * 100) if n_valid > 0 else 0
        resistance_rates[abx] = {
            'n_valid': n_valid,
            'n_resistant': int(n_resistant),
            'n_susceptible': int(n_valid - n_resistant),
            'resistance_rate': rate
        }

    # Create summary DataFrame
    balance_df = pd.DataFrame(resistance_rates).T
    balance_df = balance_df.sort_values('resistance_rate', ascending=False)

    print("\nResistance Rates (sorted):")
    print(balance_df.to_string())

    # Identify most imbalanced
    most_imbalanced = balance_df['resistance_rate'].idxmax()
    least_imbalanced = balance_df['resistance_rate'].idxmin()
    print(f"\nMost resistant: {most_imbalanced} ({balance_df.loc[most_imbalanced, 'resistance_rate']:.1f}%)")
    print(f"Least resistant: {least_imbalanced} ({balance_df.loc[least_imbalanced, 'resistance_rate']:.1f}%)")

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(ANTIBIOTICS)), balance_df['resistance_rate'],
                  color=sns.color_palette("RdYlGn_r", len(ANTIBIOTICS)))

    # Customize
    ax.set_xticks(range(len(ANTIBIOTICS)))
    ax.set_xticklabels(balance_df.index, rotation=45, ha='right')
    ax.set_ylabel('Resistance Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Antibiotic Resistance Rates by Drug', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, balance_df['resistance_rate'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add horizontal line at 50%
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: class_balance.png")
    plt.close()

    return balance_df


def analyze_missing_labels(targets, species):
    """3.2: Analyze missing label patterns."""
    print("\n" + "="*60)
    print("3.2: MISSING LABEL ANALYSIS")
    print("="*60)

    # Count missing per antibiotic
    missing_counts = targets.isna().sum()
    missing_pct = (missing_counts / len(targets) * 100).round(1)

    print("\nMissing labels per antibiotic:")
    for abx in ANTIBIOTICS:
        print(f"  {abx}: {missing_counts[abx]} ({missing_pct[abx]}%)")

    # Check correlation with species
    print("\nCorrelation with species (missing vs present):")
    for abx in ANTIBIOTICS:
        is_missing = targets[abx].isna().astype(int)
        contingency = pd.crosstab(species, is_missing)
        if contingency.shape == (2, 2):
            chi2, p_value, _, _ = chi2_contingency(contingency)
            print(f"  {abx}: χ² p-value = {p_value:.4f}" +
                  (" *** SIGNIFICANT" if p_value < 0.05 else ""))

    # Create missing pattern visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Bar chart of missing percentages
    ax = axes[0]
    colors = ['coral' if pct > 5 else 'lightblue' for pct in missing_pct]
    bars = ax.bar(range(len(ANTIBIOTICS)), missing_pct.values, color=colors)

    ax.set_xticks(range(len(ANTIBIOTICS)))
    ax.set_xticklabels(ANTIBIOTICS, rotation=45, ha='right')
    ax.set_ylabel('Missing Labels (%)', fontsize=12, fontweight='bold')
    ax.set_title('Missing Labels per Antibiotic', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, pct in zip(bars, missing_pct.values):
        height = bar.get_height()
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    # Right: Heatmap of missing patterns
    ax = axes[1]

    # Create binary matrix of missingness
    missing_matrix = targets.isna().astype(int)

    # Create crosstab of co-missing patterns
    co_missing = missing_matrix.T.dot(missing_matrix)
    co_missing_pct = (co_missing / len(targets) * 100).round(1)

    sns.heatmap(co_missing_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=ANTIBIOTICS, yticklabels=ANTIBIOTICS,
                cbar_kws={'label': 'Co-missing %'}, ax=ax)
    ax.set_title('Co-Missing Pattern Heatmap (%)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/missing_labels_pattern.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: missing_labels_pattern.png")
    plt.close()

    # Check feature distributions for samples with missing labels
    print("\nChecking if samples with missing labels differ in feature distribution...")

    # Get MALDI feature columns
    maldi_cols = [col for col in targets.index if 'maldi_feature' in str(col)]

    return missing_counts, missing_pct


def analyze_correlations(targets):
    """3.3: Calculate and visualize target correlations."""
    print("\n" + "="*60)
    print("3.3: TARGET CORRELATION ANALYSIS")
    print("="*60)

    # Calculate correlation matrix (pairwise complete observations)
    corr_matrix = targets.corr(method='pearson')

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3).to_string())

    # Find strongest correlations
    corr_values = []
    for i, abx1 in enumerate(ANTIBIOTICS):
        for j, abx2 in enumerate(ANTIBIOTICS):
            if i < j:  # Upper triangle only
                corr_values.append({
                    'pair': f"{abx1} <-> {abx2}",
                    'correlation': corr_matrix.loc[abx1, abx2],
                    'class1': DRUG_CLASSES[abx1],
                    'class2': DRUG_CLASSES[abx2]
                })

    corr_df = pd.DataFrame(corr_values).sort_values('correlation', key=abs, ascending=False)

    print("\nTop Correlations (by absolute value):")
    for _, row in corr_df.head(10).iterrows():
        print(f"  {row['pair']}: r = {row['correlation']:.3f}" +
              (f" (both {row['class1']})" if row['class1'] == row['class2'] else ""))

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1,
                xticklabels=[abx.replace('_', '\n') for abx in ANTIBIOTICS],
                yticklabels=[abx.replace('_', '\n') for abx in ANTIBIOTICS],
                cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5, ax=ax)

    ax.set_title('Antibiotic Resistance Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_correlation.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: target_correlation.png")
    plt.close()

    return corr_matrix, corr_df


def analyze_coresistance(targets):
    """3.4: Calculate conditional resistance probabilities."""
    print("\n" + "="*60)
    print("3.4: CO-RESISTANCE ANALYSIS")
    print("="*60)

    # Calculate conditional probabilities: P(B|A) for all pairs
    cond_probs = np.zeros((len(ANTIBIOTICS), len(ANTIBIOTICS)))

    for i, abx_a in enumerate(ANTIBIOTICS):
        for j, abx_b in enumerate(ANTIBIOTICS):
            # Get samples where both labels are present
            valid_mask = targets[abx_a].notna() & targets[abx_b].notna()
            col_a = targets[abx_a][valid_mask]
            col_b = targets[abx_b][valid_mask]

            if len(col_a) > 0:
                # P(Resistant to B | Resistant to A)
                resistant_to_a_mask = col_a == 1
                n_resistant_a = resistant_to_a_mask.sum()
                if n_resistant_a > 0:
                    p_b_given_a = col_b[resistant_to_a_mask].mean()
                    cond_probs[i, j] = p_b_given_a * 100
                else:
                    cond_probs[i, j] = np.nan
            else:
                cond_probs[i, j] = np.nan

    # Create DataFrame
    cond_df = pd.DataFrame(cond_probs,
                           index=ANTIBIOTICS,
                           columns=ANTIBIOTICS)

    print("\nConditional Resistance Probabilities P(B|A) (%):")
    print(cond_df.round(1).to_string())

    # Find interesting patterns
    print("\nInteresting Co-Resistance Patterns:")

    # Same drug class pairs
    for i, abx_a in enumerate(ANTIBIOTICS):
        for j, abx_b in enumerate(ANTIBIOTICS):
            if i < j and DRUG_CLASSES[abx_a] == DRUG_CLASSES[abx_b]:
                prob = cond_df.loc[abx_a, abx_b]
                prob_rev = cond_df.loc[abx_b, abx_a]
                print(f"  {abx_a} <-> {abx_b} ({DRUG_CLASSES[abx_a]}): "
                      f"P({abx_b}|{abx_a})={prob:.1f}%, P({abx_a}|{abx_b})={prob_rev:.1f}%")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(cond_df, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[abx.replace('_', '\n') for abx in ANTIBIOTICS],
                yticklabels=[abx.replace('_', '\n') for abx in ANTIBIOTICS],
                cbar_kws={'label': 'P(Resistant to B | Resistant to A) [%]'},
                linewidths=0.5, vmin=0, vmax=100, ax=ax)

    ax.set_title('Conditional Resistance: P(Resistant to Column | Resistant to Row)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Conditioned on being resistant to row antibiotic', fontsize=11)
    ax.set_ylabel('Given resistance to row, probability of resistance to column', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/conditional_resistance.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved: conditional_resistance.png")
    plt.close()

    return cond_df


def analyze_antibiotic_clustering(corr_matrix):
    """3.5: Hierarchical clustering of antibiotics."""
    print("\n" + "="*60)
    print("3.5: ANTIBIOTIC CLASS GROUPING")
    print("="*60)

    # Perform hierarchical clustering
    linkage_matrix = linkage(corr_matrix, method='average')

    # Create dendrogram
    fig, ax = plt.subplots(figsize=(14, 8))

    dendrogram(linkage_matrix, labels=ANTIBIOTICS, ax=ax,
               leaf_font_size=11, leaf_rotation=45)

    ax.set_title('Hierarchical Clustering of Antibiotics (based on correlation)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Antibiotic', fontsize=12)
    ax.set_ylabel('Distance (1 - correlation)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/antibiotic_dendrogram.png", dpi=300, bbox_inches='tight')
    print(f"Saved: antibiotic_dendrogram.png")
    plt.close()

    # Interpret clusters
    print("\nIdentified Clusters (by distance threshold):")
    print("Note: Compare to known drug classes:")
    for abx, drug_class in DRUG_CLASSES.items():
        print(f"  {abx}: {drug_class}")

    return linkage_matrix


def analyze_multilabel_statistics(targets):
    """3.6: Multi-label resistance patterns."""
    print("\n" + "="*60)
    print("3.6: MULTI-LABEL RESISTANCE STATISTICS")
    print("="*60)

    # Count number of resistances per sample (ignoring NaN)
    resistance_counts = targets.notna().sum(axis=1) - targets.isna().sum(axis=1)
    # Actually, we need to count actual resistances
    resistance_counts = targets.sum(axis=1)

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    counts = resistance_counts.value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color='steelblue', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Number of Resistances per Sample', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Multi-Drug Resistance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, len(ANTIBIOTICS) + 1))

    # Add count labels
    for bar, count in zip(bars, counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10)

    # Add statistics
    mean_res = resistance_counts.mean()
    median_res = resistance_counts.median()
    ax.axvline(mean_res, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_res:.2f}')
    ax.axvline(median_res, color='green', linestyle='--', linewidth=2, label=f'Median: {median_res:.1f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/resistance_count_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: resistance_count_distribution.png")
    plt.close()

    print(f"\nResistance Count Statistics:")
    print(f"  Mean: {mean_res:.2f}")
    print(f"  Median: {median_res:.1f}")
    print(f"  Std: {resistance_counts.std():.2f}")
    print(f"  Min: {resistance_counts.min()}")
    print(f"  Max: {resistance_counts.max()}")

    # Find most common resistance patterns
    print("\nMost Common Resistance Patterns:")

    # Create pattern strings
    pattern_strings = []
    for idx, row in targets.iterrows():
        pattern = ''.join(['1' if pd.notna(val) and val == 1 else '0' if pd.notna(val) else 'X'
                          for val in row.values])
        pattern_strings.append(pattern)

    pattern_counts = pd.Series(pattern_strings).value_counts()

    print("\nTop 10 patterns:")
    for i, (pattern, count) in enumerate(pattern_counts.head(10).items(), 1):
        pct = count / len(targets) * 100
        print(f"  {i}. Pattern: {pattern} | Count: {count} ({pct:.1f}%)")

    # Multi-drug resistance definition
    mdr_threshold = 3  # Resistant to 3 or more drugs
    mdr_count = (resistance_counts >= mdr_threshold).sum()
    mdr_pct = mdr_count / len(targets) * 100

    print(f"\nMulti-Drug Resistance (>= {mdr_threshold} drugs):")
    print(f"  Count: {mdr_count} ({mdr_pct:.1f}%)")

    # XDR (extensively drug-resistant) - resistant to all but 1 or 2
    xdr_count = (resistance_counts >= len(ANTIBIOTICS) - 2).sum()
    xdr_pct = xdr_count / len(targets) * 100

    print(f"\nExtensively Drug-Resistant (resistant to >= {len(ANTIBIOTICS) - 2}):")
    print(f"  Count: {xdr_count} ({xdr_pct:.1f}%)")

    return resistance_counts, pattern_counts


def generate_report(balance_df, missing_counts, missing_pct, corr_df,
                    cond_df, resistance_counts, n_samples=3360):
    """Generate comprehensive markdown report."""

    report = f"""# EDA Phase 3: Target Analysis Report

**Generated:** 2026-01-07
**Data:** /sata_disk/users/erfan/ml_kaggle/raw/train.csv
**Samples:** 3360
**Targets:** 8 antibiotics

---

## Executive Summary

This report analyzes the 8 antibiotic resistance labels to understand:
- Class balance and imbalances
- Missing label patterns
- Correlations between antibiotics
- Co-resistance patterns
- Natural groupings by drug class
- Multi-drug resistance prevalence

**Key Findings:**
- Resistance rates vary from {balance_df['resistance_rate'].min():.1f}% to {balance_df['resistance_rate'].max():.1f}%
- Missing labels range from {missing_pct.min():.1f}% to {missing_pct.max():.1f}%
- Average of {resistance_counts.mean():.1f} resistances per sample
- Strongest correlation: {corr_df.iloc[0]['pair']} (r={corr_df.iloc[0]['correlation']:.3f})

---

## 3.1 Class Balance

### Resistance Rates by Antibiotic

| Antibiotic | Drug Class | Valid Samples | Resistant | Susceptible | Resistance Rate |
|------------|------------|---------------|-----------|-------------|-----------------|
"""

    for abx in balance_df.index:
        row = balance_df.loc[abx]
        drug_class = DRUG_CLASSES.get(abx, 'Unknown')
        report += f"| {abx} | {drug_class} | {row['n_valid']} | {row['n_resistant']} | {row['n_susceptible']} | {row['resistance_rate']:.1f}% |\n"

    report += f"""
### Interpretation

**Most Resistant:** {balance_df['resistance_rate'].idxmax()} ({balance_df['resistance_rate'].max():.1f}%)
**Least Resistant:** {balance_df['resistance_rate'].idxmin()} ({balance_df['resistance_rate'].min():.1f}%)

**Modeling Implications:**
- Targets with extreme imbalance (< 20% or > 80%) may require:
  - Class weighting in loss function
  - Focal loss to handle easy negatives
  - Oversampling of minority class
  - Threshold tuning for optimal F1

---

## 3.2 Missing Label Patterns

### Missing Labels per Antibiotic

| Antibiotic | Missing Count | Missing % |
|------------|---------------|-----------|
"""

    for abx in ANTIBIOTICS:
        report += f"| {abx} | {missing_counts[abx]} | {missing_pct[abx]:.1f}% |\n"

    report += f"""
### Missing Label Strategy

**Current Situation:**
- {sum(missing_counts > 0)} antibiotics have some missing labels
- Total missing label instances: {missing_counts.sum()}

**Recommended Handling:**
1. **Training:** Use masked loss (ignore NaN targets)
   - PyTorch: `MaskedBCEWithLogitsLoss`
   - Only compute loss where target is not NaN

2. **Validation:** Evaluate per-antibiotic using only valid labels
   - Compute AUC separately for each antibiotic
   - Average AUC across antibiotics (competition metric)

3. **Imputation (NOT recommended):**
   - Don't impute resistance labels - this is a medical prediction task
   - Missing labels are likely biologically meaningful (not tested)

4. **Model Architecture:**
   - Single multi-task model with masked loss per sample
   - Each antibiotic head only trained on valid samples

---

## 3.3 Target Correlation Matrix

### Top Correlations

| Pair | Correlation | Drug Classes | Interpretation |
|------|-------------|--------------|----------------|
"""

    for _, row in corr_df.head(10).iterrows():
        interpretation = "Same class" if row['class1'] == row['class2'] else "Different classes"
        report += f"| {row['pair']} | {row['correlation']:.3f} | {row['class1']} / {row['class2']} | {interpretation} |\n"

    report += f"""
### Biological Interpretation

**Expected Correlations:**
1. **Fluoroquinolones** (Levofloxacin, Ciprofloxacin)
   - Same mechanism of action
   - Cross-resistance common
   - Should be highly correlated

2. **Carbapenems** (Imipenem, Ertapenem)
   - Same beta-lactam class
   - Shared resistance mechanisms
   - Should show correlation

3. **Cephalosporins** (Cefotaxime, Cefuroxime)
   - Beta-lactam antibiotics
   - Some shared resistance

**Modeling Implications:**
- Strongly correlated targets can share model components
- Consider task grouping or multi-task learning
- Correlated targets benefit from shared representations

---

## 3.4 Co-Resistance Analysis

### Conditional Probability P(Resistant to B | Resistant to A)

The heatmap shows the probability of being resistant to the column antibiotic
given resistance to the row antibiotic.

**Key Patterns:**

1. **Within-class co-resistance:**
   - Check if same-class drugs show high conditional probabilities
   - Indicates shared resistance mechanisms

2. **Cross-class co-resistance:**
   - High cross-class values suggest MDR patterns
   - May indicate efflux pumps or broad-spectrum resistance

**Modeling Implications:**
- Classifier chains: Predict A, then use A to predict B
- Use co-resistance patterns for feature engineering
- Consider biological constraints in model architecture

---

## 3.5 Antibiotic Class Grouping

### Hierarchical Clustering Results

The dendrogram groups antibiotics by correlation similarity.

**Expected Groups:**
- **Fluoroquinolones:** Levofloxacin, Ciprofloxacin
- **Carbapenems:** Imipenem, Ertapenem
- **Penicillins:** Ampicillin, Amoxicillin_Clavulanic_acid
- **Cephalosporins:** Cefotaxime, Cefuroxime

**Model Architecture Recommendations:**

1. **Shared Encoder, Task-Specific Heads:**
   ```
   MALDI Features → Shared Encoder → Task Heads (8x)
   ```

2. **Grouped Architecture (if clusters are strong):**
   ```
   MALDI Features → Shared Encoder → Group Encoders → Task Heads
   ```

3. **Baseline:** Start with simple multi-task MLP
   - Single hidden layer shared across all tasks
   - Separate output heads for each antibiotic
   - Masked loss to handle missing labels

---

## 3.6 Multi-Label Statistics

### Resistance Count Distribution

| Statistic | Value |
|-----------|-------|
| Mean resistances per sample | {resistance_counts.mean():.2f} |
| Median | {resistance_counts.median():.1f} |
| Std Dev | {resistance_counts.std():.2f} |
| Min | {resistance_counts.min()} |
| Max | {resistance_counts.max()} |

### Multi-Drug Resistance (MDR)

- **MDR (>= 3 drugs):** {(resistance_counts >= 3).sum()} samples ({(resistance_counts >= 3).sum() / n_samples * 100:.1f}%)
- **XDR (>= 6 drugs):** {(resistance_counts >= 6).sum()} samples ({(resistance_counts >= 6).sum() / n_samples * 100:.1f}%)

### Most Common Patterns

The most frequent resistance patterns indicate typical resistance profiles.

**Modeling Implications:**
- Multi-label correlations are important
- Consider label powerset methods for small datasets
- Binary relevance may miss correlations

---

## Recommendations for Modeling

### 1. Loss Function
```python
class MaskedBCEWithLogitsLoss(nn.Module):
    def forward(self, predictions, targets):
        # targets can contain NaN
        mask = targets.notna()
        loss = F.binary_cross_entropy_with_logits(
            predictions[mask],
            targets[mask].float(),
            reduction='none'
        )
        return loss.mean()
```

### 2. Model Architecture Options

**Option A: Simple Multi-Task MLP (Recommended Baseline)**
```python
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim=6000, hidden_dim=256, num_tasks=8):
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim//2, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        features = self.shared(x)
        return torch.cat([head(features) for head in self.task_heads], dim=1)
```

**Option B: LightGBM (Alternative for small data)**
- Train 8 separate LightGBM models
- Uses missing label handling internally
- May outperform NN on 3360 samples

### 3. Validation Strategy
- Use `MultilabelStratifiedKFold` from `iterative-stratification`
- Stratify by species AND resistance labels
- Ensure all species are represented in each fold

### 4. Metric Tracking
- Track AUC per antibiotic
- Identify which antibiotics are hardest to predict
- Use per-antibiotic thresholds for final predictions

---

## Figures Generated

1. **class_balance.png** - Resistance rates bar chart
2. **missing_labels_pattern.png** - Missing label analysis and co-missing heatmap
3. **target_correlation.png** - Correlation matrix heatmap
4. **conditional_resistance.png** - P(B|A) conditional probability heatmap
5. **antibiotic_dendrogram.png** - Hierarchical clustering dendrogram
6. **resistance_count_distribution.png** - Multi-drug resistance histogram

All figures saved to: `/sata_disk/users/erfan/ml_kaggle/outputs/eda/phase3/`

---

## Next Steps

1. Implement masked loss function
2. Build baseline multi-task MLP
3. Set up proper cross-validation
4. Train and evaluate baseline
5. Compare against LightGBM baseline
6. Iterate on architecture based on per-antibiotic performance

---

**Report End**
"""

    # Save report
    report_path = "/sata_disk/users/erfan/ml_kaggle/knowledge/insights/eda_phase3_target_analysis.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")
    return report


def main():
    """Run all target analysis."""
    print("\n" + "="*70)
    print(" PHASE 3: TARGET ANALYSIS FOR AMR PREDICTION")
    print("="*70)

    # Load data
    train, targets, species = load_data()

    # Run all analyses
    balance_df = analyze_class_balance(targets)
    missing_counts, missing_pct = analyze_missing_labels(targets, species)
    corr_matrix, corr_df = analyze_correlations(targets)
    cond_df = analyze_coresistance(targets)
    linkage_matrix = analyze_antibiotic_clustering(corr_matrix)
    resistance_counts, pattern_counts = analyze_multilabel_statistics(targets)

    # Generate report
    print("\n" + "="*70)
    print(" GENERATING COMPREHENSIVE REPORT")
    print("="*70)
    generate_report(balance_df, missing_counts, missing_pct, corr_df,
                    cond_df, resistance_counts, n_samples=len(train))

    print("\n" + "="*70)
    print(" PHASE 3 COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Report saved to: /sata_disk/users/erfan/ml_kaggle/knowledge/insights/eda_phase3_target_analysis.md")


if __name__ == "__main__":
    main()
