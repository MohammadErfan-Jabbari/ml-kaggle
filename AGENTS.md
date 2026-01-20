## Repository: AMR Prediction from MALDI-TOF (Kaggle)

### Session management (poster-focused, across sessions)
- **Single source of truth**: `knowledge/POSTER_STATE.md` (current storyline, figures, numbers, decisions).
- **Checklist**: `knowledge/POSTER_TODO.md` (what’s left before print).
- **Per-session notes**: add a short dated note under `knowledge/sessions/` named `YYYY-MM-DD_poster.md` and link it from `knowledge/POSTER_STATE.md`.
- **Start of each session**
  1) Read `AGENTS.md` and `CLAUDE.md`
  2) Read `knowledge/POSTER_STATE.md` and `knowledge/POSTER_TODO.md`
  3) Pick 1–3 tasks for this session (update `knowledge/POSTER_TODO.md`)
- **End of each session**
  1) Update `knowledge/POSTER_STATE.md` (what changed + decision log)
  2) Update `knowledge/POSTER_TODO.md` (check off items)
  3) If LaTeX poster changed, ensure `poster/main.tex` still compiles locally.

### Competition context (from provided Kaggle PDF)
- Task: predict resistance (0/1) for **8 antibiotics** per sample (multi-label).
- Setting: **semi-supervised** (some training labels missing / partially labeled).
- Evaluation: **mean AUC across the 8 antibiotics** (public LB uses 40% of test, private 60%).
- Course grading reference points: AUC 0.50 → score 0, AUC 0.80 → score 4, best team → score 10; poster session adds separate grade component.

### Project status (repo reality)
- Best recorded public LB: **0.83862** via **mega-blend (rank averaging)**.
- Final private leaderboard screenshot indicates **rank improvement +4** (private vs public) for both the individual and team entries.
- Biggest pitfalls found:
  - **Wrong metric optimization** (optimizing K. pneumoniae AUC hurts mean-AUC LB).
  - **Stacking overfits** (very high OOF but lower LB); prefer **rank averaging**.
  - **Species distribution shift** is critical; validation must reflect test distribution.
  - **Missing labels** are not random (especially Amoxicillin/Clavulanic acid); always mask NaNs.

### “Read-first” files
- `README.md` (quick start + best LB + repo map)
- `knowledge/SESSION_STATE.md` (current best, what works/doesn’t)
- `knowledge/EDA_CONCLUSIONS_STRATEGY.md` (5 key truths)
- `knowledge/submissions/submissions_log.md` (what was submitted + LB)
- `knowledge/sessions/2026-01-08-session-handoff.md` (end-of-project run recap)
- `knowledge/POSTER_STATE.md` (poster storyline + assets)
- `knowledge/POSTER_TODO.md` (poster checklist)

### Key implementation hooks
- Validation split that matches test species distribution:
  - `src/data/dataset.py` → `load_validation_split()`
- Intrinsic resistance rules (biological heuristics used in modeling/inference):
  - Implemented in several experiment scripts (e.g. `experiments/run_mega_blend.py`)

### Main runnable scripts (common entrypoints)
- Best known submission recipe:
  - `uv run python experiments/run_mega_blend.py`
- Semi-supervised approach:
  - `uv run python experiments/run_self_training.py`
- Large ensemble (did not beat best in logs, but exists):
  - `uv run python experiments/run_miracle_v2.py`
- EDA reproduction:
  - `uv run python scripts/eda/phase1_data_profiling.py` (and phase2–phase8)

### Outputs to cite / reuse
- Kaggle-ready submissions: `outputs/submissions/`
- Experiment result summaries: `outputs/experiments/`
- Run artifacts: `outputs/*_runs/` (self-training, miracle_v2, blends, transductive DR, etc.)
- Poster source (LaTeX, A1): `poster/main.tex`
- Poster PDF (A1): `poster/main.pdf` (verify with `pdfinfo poster/main.pdf`)
  - Current orientation: portrait (A1 = 59.4 × 84.1 cm)

### Repo gotchas
- Some scripts use absolute paths (e.g. `/home/erfan/data/ml_kaggle/raw`, `/sata_disk/.../outputs`).
  - If running on a different machine, patch these to use repo-relative paths.
