# CLAUDE.md — AMR Prediction from MALDI-TOF (Kaggle)

Authoritative session guide for this repo. Self-contained: everything a fresh
session needs is here. Deeper detail lives in the linked files under `knowledge/`.

---

## 1. What this repo is

Work for the Kaggle competition **Antimicrobial Resistance (AMR) Prediction from
MALDI-TOF**: predict bacterial resistance (0/1) to **8 antibiotics** per sample
from MALDI-TOF mass-spectrometry features. The active deliverable now is the
**academic poster** (`poster/`); the modeling competition is effectively done.

- **Task**: multi-label classification (8 antibiotics), **semi-supervised** (labels partially missing).
- **Metric**: **mean AUC across the 8 antibiotics** (public LB = 40% of test, private = 60%).
- **Best result**: public LB **0.83862** via **mega-blend (rank averaging)**; private LB rank improved **+4** vs public.
- **Grading reference**: AUC 0.50 → 0, 0.80 → 4, best team → 10; poster is a separate grade component.

---

## 2. Environment & how to run

This repo was copied from another machine and adopted here on **2026-06-02**
(ownership taken by `centcom`, permissions normalized, `.venv` rebuilt, all
hardcoded foreign paths remapped to the repo root). It is now fully runnable.

- **Repo root**: `/home/centcom/data/ml_kaggle`
- **Package manager**: `uv` (Python ≥ 3.11). Entrypoint is always `uv run python <script>`.
- **Rebuild env** (if `.venv` is ever broken again): `uv sync`
- **Smoke test** (fast end-to-end sanity check — run this to verify the repo works):
  ```bash
  uv run python scripts/smoke_test.py     # should end with "✅ All smoke tests passed!"
  ```
- **Data** lives in `raw/` (gitignored, present locally): `train.csv` (96 MB),
  `test.csv` (28 MB), `sample_submission.csv`, `species_mapping.csv`.
  Do **not** `Read` the big CSVs into context — inspect via code (`uv run python ...`).

### Main entrypoints
```bash
uv run python experiments/run_mega_blend.py      # best submission recipe (LB 0.83862)
uv run python experiments/run_self_training.py   # semi-supervised approach (LB 0.82445)
uv run python experiments/run_miracle_v2.py       # 17-model ensemble (did not beat best)
uv run python scripts/eda/phase1_data_profiling.py   # EDA (phases 1–8 exist)
```

### Submit to Kaggle
```bash
kaggle competitions submit -c antimicrobial-resistance-prediction-from-maldi-tof \
  -f outputs/submissions/<file>.csv -m "message"
```

---

## 3. The data

| Aspect | Details |
|--------|---------|
| Input | 6,000 MALDI-TOF spectral features (binned m/z, ~3 Da per bin), **93.3% zeros** |
| Output | binary resistance for 8 antibiotics |
| Samples | 3,360 train, 1,000 test |
| Species | 4 bacterial species, **strong train↔test distribution shift** |
| Labels | partially missing (42.8% missing for Amoxicillin/Clavulanic acid); missingness is **not random** |

**8 target antibiotics** (with train resistance rate): Ampicillin 88.2%,
Amoxicillin_Clavulanic_acid 29.9%, Cefotaxime 73.8%, Cefuroxime 76.4%,
Ciprofloxacin 61.1%, Ertapenem 64.5%, Imipenem 38.9%, Levofloxacin 62.4%.

---

## 4. Critical insights (read before modeling)

These are the hard-won truths. Violating them is how earlier attempts failed.
Full version: `knowledge/EDA_CONCLUSIONS_STRATEGY.md`.

1. **Species distribution shift is the whole game.** Validation MUST match the
   test species distribution, or scores lie.

   | Species (id) | Train % | Test % | Weight |
   |---|---|---|---|
   | P. aeruginosa (3) | 43.1% | 3.0% | **down ~0.05–0.1×** |
   | K. pneumoniae (1) | 27.9% | 50.8% | **up ~2–3×** |
   | E. coli (0) | 16.6% | 26.9% | up ~1.5× |
   | P. mirabilis (2) | 12.4% | 19.3% | up ~1.5× |

2. **Intrinsic resistance = free, deterministic predictions** (~18.8% of test):
   - P. aeruginosa → always resistant to Ampicillin, Amox/Clav, Ertapenem, Cefotaxime, Cefuroxime.
   - P. mirabilis → always resistant to Imipenem.

3. **OOF predictions are unreliable** (~10% optimistic). Trust the
   distribution-matched validation split, not OOF.

4. **Optimize the right metric.** Optimizing K. pneumoniae AUC *alone* hurts
   mean-AUC LB. Always optimize the masked mean AUC across all 8.

5. **Label correlations** (shared resistance mechanisms — useful for multi-task):
   Levofloxacin↔Ciprofloxacin r=0.925, Imipenem↔Ertapenem r=0.772, Ertapenem↔Cefotaxime r=0.813.

### What works vs what doesn't
| Works | Doesn't |
|---|---|
| **Rank averaging** ensembles | **Stacking** (~12.8% overfit: high OOF, low LB) |
| Model diversity (LGB + XGB + CatBoost + MLP) | Neural nets alone |
| LightGBM on sparse spectra | Unsupervised DR (PCA, KPCA) |
| **PLS** (supervised) dimensionality reduction | Optimizing one species' AUC |
| Always masking NaN labels | Treating missing labels as random |

### Best model: mega-blend
Rank-average of 3 diverse pipelines — **PLS-LGB**, **Species-Global-Blend**,
**Tuned-LGB** — then apply species reweighting + intrinsic-resistance rules.

---

## 5. Key code hooks

- **Validation split matching test distribution** (use this, not a random split):
  ```python
  from src.data.dataset import load_validation_split
  X_train, X_val, y_train, y_val, species_train, species_val = load_validation_split()
  # Train: 2688, Val: 672 (matches test species distribution)
  ```
- **Masked mean AUC** (handles NaN labels): `from src.utils.metrics import mean_auc`
- **Intrinsic resistance rules**: implemented inline in experiment scripts (e.g.
  `experiments/run_mega_blend.py`); apply by species id after model prediction.

---

## 6. Repo map

```
raw/                  competition data (gitignored, present locally)
data/processed/       validation-split cache
src/                  data/ features/ models/ training/ inference/ utils/
experiments/          30+ run scripts (run_mega_blend.py = best)
scripts/eda/          EDA generation, phases 1–8
outputs/              submissions/  experiments/  eda/  *_runs/
knowledge/            all docs, insights, sessions, hypotheses, submissions log
poster/               LaTeX A1 poster (main.tex), build.sh → main.pdf + main.png
```

---

## 7. Poster workflow (current active work)

- Source: `poster/main.tex` — build with `./poster/build.sh` (generates
  `poster/main.pdf` **and** a `poster/main.png` preview).
  **Always eyeball the PNG after building** — quickest way to catch layout breaks.
- Verify PDF: `pdfinfo poster/main.pdf` (A1 = 59.4 × 84.1 cm, currently portrait).
- State: `knowledge/POSTER_STATE.md` (storyline, figures, numbers, decision log).
- Checklist: `knowledge/POSTER_TODO.md`.

---

## 8. Session management process

**Single source of truth**: `knowledge/POSTER_STATE.md`. **Checklist**: `knowledge/POSTER_TODO.md`.

- **Start of session**: read this file → `knowledge/POSTER_STATE.md` → `knowledge/POSTER_TODO.md`; pick 1–3 tasks.
- **During**: per-session note under `knowledge/sessions/` named `YYYY-MM-DD_poster.md`, linked from `POSTER_STATE.md`.
- **End of session**: update `POSTER_STATE.md` (what changed + decisions) and `POSTER_TODO.md` (check off); if the poster changed, rebuild and sanity-check the PNG.

---

## 9. Read-first files

| File | Why |
|---|---|
| `knowledge/POSTER_STATE.md` | poster storyline + assets (current focus) |
| `knowledge/POSTER_TODO.md` | poster checklist |
| `knowledge/EDA_CONCLUSIONS_STRATEGY.md` | the 5 modeling truths (full) |
| `knowledge/SESSION_STATE.md` | current best, what works/doesn't |
| `knowledge/submissions/submissions_log.md` | what was submitted + LB |
| `knowledge/sessions/2026-01-08-session-handoff.md` | end-of-competition recap |
| `README.md` | long-form overview, results table, code patterns |

---

## 10. Gotchas

- Scripts use **absolute paths** rooted at `/home/centcom/data/ml_kaggle` (not
  repo-relative). If the repo ever moves, remap them again across `*.py/*.yaml/*.sh`.
- Session-log files (`here.txt`, `last_chat.txt`, `last_dance.txt`) still contain
  the original `/home/erfan` and `/sata_disk/...` paths — that's historical text, ignore.
- Big CSVs in `raw/` must be inspected via code, never `Read` directly.
