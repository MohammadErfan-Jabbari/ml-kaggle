# Poster State (A1, LaTeX) — AMR Kaggle Project

**Authors**: MohammadErfan Jabbari, Ana Mirza  
**Poster source**: `poster/main.tex`  
**Printing target**: A1 (standard), per instructor email (poster boards stated as 50×70 cm but A1 recommended).

## Core story (what we want the viewer to learn)
1. The project was run as an **iterative ML workflow**: EDA → hypothesis → experiment → validation → submission → reflection.
2. The biggest performance drivers were **problem-specific constraints** (species shift, missing labels, intrinsic resistance), not exotic architectures.
3. Simple, robust ensembling (**rank averaging**) beat higher-variance methods (stacking overfit).
4. Semi-supervised methods helped locally (self-training), but did not surpass the best ensemble in our runs.

## Must-include figures (from `outputs/eda/`)
- Species shift: `outputs/eda/species_distribution.png` (or `outputs/eda/phase6/species_stratified_shift.png`)
- Missingness: `outputs/eda/missing_label_heatmap.png`
- One “spectra/feature” figure: `outputs/eda/phase2/example_spectra.png` or `outputs/eda/phase2/sparsity_by_feature.png`
- (Optional) Target correlations: `outputs/eda/target_correlation.png`

## Poster build status
- `poster/main.tex` compiles successfully to `poster/main.pdf`.
- Current poster expects these files in `poster/figs/`:
  - `poster/figs/species_distribution.png`
  - `poster/figs/missing_label_heatmap.png`
  - `poster/figs/sparsity_by_feature.png`

## Results to cite (from provided screenshots)
| Submission file | Public | Private | Notes |
|---|---:|---:|---|
| `sub_mega_blend_rank_avg_20260107_2057.csv` | 0.83862 | 0.81307 | Best public; private shown in screenshot |
| `sub_final_mega_st_rank_20260108_215951.csv` | 0.83660 | 0.80938 | Self-train heavy blend; selected in screenshot |
| `kaggle_submission.csv` | 0.80223 | 0.78556 | Earlier baseline by Ana |
| `semisupervised_msdeepamr_submission.csv` | 0.70025 | 0.66517 | Deep model attempt |
| `msdeepamr_submission.csv` | 0.69924 | 0.66328 | Deep model attempt |

## Final leaderboard note (from screenshot)
- Private leaderboard (final): **rank change +4** (private better than public) shown for:
  - `MohammadErfan Jabbari` (score 0.81307)
  - `Erfan-Ana` (score 0.81307)

## Experiments worth mentioning (repo artifacts)
- Best ensemble method: mega-blend rank averaging (`experiments/run_mega_blend.py`)
- Self-training run (best in logs): `outputs/self_training_runs/run_20260108_180837/results.json` (val mean AUC ≈ 0.8179)
- Large ensemble (17 models): `outputs/miracle_v2_runs/run_20260108_180757/results.json` (best val mean AUC ≈ 0.8147)
- Species-specific models: `outputs/species_specific_runs/run_20260108_194209/results.json` (mean AUC ≈ 0.8023)
- Transductive DR attempt (PCA): `outputs/transductive_dr_runs/pca_100_20260108_211030/results.json` (val mean AUC ≈ 0.7758)

## Decisions log
- Orientation: **A1 portrait** (closer to 50×70 boards; A1 recommended for printing).
- Poster focuses on **methods explored + lessons**, not only the best submission.

## Session notes index
- Add new notes here as you work: `knowledge/sessions/YYYY-MM-DD_poster.md`
