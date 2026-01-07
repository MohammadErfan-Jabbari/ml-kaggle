# MALDI-TOF Mass Spectrometry for AMR Prediction

## Fundamentals

### How MALDI-TOF MS Works
1. **Sample Preparation**: Bacterial colonies mixed with matrix (α-cyano-4-hydroxycinnamic acid)
2. **Laser Desorption**: Pulsed laser irradiates sample, matrix absorbs energy
3. **Ionization**: Molecules ionized (typically [M+H]+)
4. **Time-of-Flight**: Ions accelerated; lighter ions travel faster
5. **Detection**: Time-to-detector correlates with mass → spectrum generated

### Spectral Features
- **Mass range**: 2,000-20,000 Da (Daltons)
- **Primary signals**: Ribosomal proteins (most abundant and conserved)
- **Peak intensities**: Relative abundance of proteins at each m/z value
- **Our data**: 6,000 binned features covering this range (~3 Da per bin)

## Preprocessing (Standard Pipeline)

| Step | Technique | Purpose |
|------|-----------|---------|
| Smoothing | Savitzky-Golay filter | Remove high-frequency noise |
| Baseline Correction | SNIP algorithm | Remove background signal drift |
| Normalization | TIC or PQN | Enable comparison across samples |
| Peak Detection | SNR thresholding | Identify significant peaks |
| Binning | Fixed-width bins (3 Da) | Create uniform feature vectors |

**Note**: Our competition data is already binned to 6000 features.

## Key Peaks for AMR Detection

### S. aureus (MRSA)
| m/z Peak | Associated With |
|----------|-----------------|
| 2,414 Da | PSM-mec (MRSA marker) |
| 3,006 Da | agr-positive strains |
| 4,517 Da | Clonal complex CC398 |

### E. coli
| m/z Peak | Associated With |
|----------|-----------------|
| 8,450 Da | Multi-drug resistance |
| 6,800-6,900 Da | Resistance differentiation |

### K. pneumoniae
- Peaks in 2,000-3,000 Da range most discriminative
- 7,770, 4,736, 2,135, 7,706 Da associated with resistance

**SHAP Analysis Finding**: The 2,000-7,000 Da range (lower mass region, features 0-1666) contains most discriminative features for AMR.

## State-of-the-Art Performance

| Study | Method | AUROC Range |
|-------|--------|-------------|
| DRIAMS (2022) | LightGBM/MLP | 0.74-0.80 |
| MSDeepAMR (2024) | 1D CNN | 0.82-0.93 |
| P. aeruginosa (2023) | MLP | up to 0.87 |
| Maldi Transformer (2025) | Self-supervised | Improved over baselines |

## Key Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| Site-specific drift | Models trained at one hospital perform worse at others | Transfer learning, domain adaptation |
| Temporal degradation | Performance decreases over time | Regular retraining, temporal validation |
| Class imbalance | Resistant samples often minority class | Class weighting, AUPRC metric |
| Indirect detection | MALDI detects proteins, not resistance genes | Focus on correlated protein markers |

## Key References

- **DRIAMS Paper**: https://www.nature.com/articles/s41591-021-01619-9
- **DRIAMS Code**: https://github.com/BorgwardtLab/maldi_amr
- **maldi-learn**: https://github.com/BorgwardtLab/maldi-learn
- **MSDeepAMR**: https://www.frontiersin.org/articles/10.3389/fmicb.2024.1361795
