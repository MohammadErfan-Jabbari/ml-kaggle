# EDA Phase 3: Target Analysis Report

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
- Resistance rates vary from 29.9% to 88.2%
- Missing labels range from 0.0% to 42.8%
- Average of 4.4 resistances per sample
- Strongest correlation: Levofloxacin <-> Ciprofloxacin (r=0.925)

---

## 3.1 Class Balance

### Resistance Rates by Antibiotic

| Antibiotic | Drug Class | Valid Samples | Resistant | Susceptible | Resistance Rate |
|------------|------------|---------------|-----------|-------------|-----------------|
| Ampicillin | Penicillin | 3332.0 | 2938.0 | 394.0 | 88.2% |
| Cefuroxime | Cephalosporin (2nd gen) | 3327.0 | 2502.0 | 825.0 | 75.2% |
| Cefotaxime | Cephalosporin (3rd gen) | 3357.0 | 1886.0 | 1471.0 | 56.2% |
| Imipenem | Carbapenem | 3249.0 | 1802.0 | 1447.0 | 55.5% |
| Ciprofloxacin | Fluoroquinolone | 3274.0 | 1747.0 | 1527.0 | 53.4% |
| Levofloxacin | Fluoroquinolone | 3271.0 | 1709.0 | 1562.0 | 52.2% |
| Ertapenem | Carbapenem | 3360.0 | 1557.0 | 1803.0 | 46.3% |
| Amoxicillin_Clavulanic_acid | Penicillin+BetaLactamase | 1921.0 | 575.0 | 1346.0 | 29.9% |

### Interpretation

**Most Resistant:** Ampicillin (88.2%)
**Least Resistant:** Amoxicillin_Clavulanic_acid (29.9%)

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
| Ampicillin | 28 | 0.8% |
| Levofloxacin | 89 | 2.6% |
| Ciprofloxacin | 86 | 2.6% |
| Imipenem | 111 | 3.3% |
| Amoxicillin_Clavulanic_acid | 1439 | 42.8% |
| Ertapenem | 0 | 0.0% |
| Cefotaxime | 3 | 0.1% |
| Cefuroxime | 33 | 1.0% |

### Missing Label Strategy

**Current Situation:**
- 7 antibiotics have some missing labels
- Total missing label instances: 1789

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
| Levofloxacin <-> Ciprofloxacin | 0.925 | Fluoroquinolone / Fluoroquinolone | Same class |
| Ertapenem <-> Cefotaxime | 0.813 | Carbapenem / Cephalosporin (3rd gen) | Different classes |
| Imipenem <-> Ertapenem | 0.772 | Carbapenem / Carbapenem | Same class |
| Cefotaxime <-> Cefuroxime | 0.657 | Cephalosporin (3rd gen) / Cephalosporin (2nd gen) | Different classes |
| Levofloxacin <-> Cefotaxime | 0.614 | Fluoroquinolone / Cephalosporin (3rd gen) | Different classes |
| Imipenem <-> Cefotaxime | 0.592 | Carbapenem / Cephalosporin (3rd gen) | Different classes |
| Ciprofloxacin <-> Cefotaxime | 0.575 | Fluoroquinolone / Cephalosporin (3rd gen) | Different classes |
| Ertapenem <-> Cefuroxime | 0.538 | Carbapenem / Cephalosporin (2nd gen) | Different classes |
| Amoxicillin_Clavulanic_acid <-> Cefuroxime | 0.520 | Penicillin+BetaLactamase / Cephalosporin (2nd gen) | Different classes |
| Levofloxacin <-> Ertapenem | 0.494 | Fluoroquinolone / Carbapenem | Different classes |

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
| Mean resistances per sample | 4.38 |
| Median | 5.0 |
| Std Dev | 2.46 |
| Min | 0.0 |
| Max | 8.0 |

### Multi-Drug Resistance (MDR)

- **MDR (>= 3 drugs):** 2304 samples (68.6%)
- **XDR (>= 6 drugs):** 1410 samples (42.0%)

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
