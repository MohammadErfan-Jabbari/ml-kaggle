#!/usr/bin/env python3
"""Generate final blend candidate submission CSVs.

This script is intentionally training-free: it blends existing submission
CSVs that are already produced by prior runs.

Inputs (expected to exist):
- outputs/submissions/sub_mega_blend_rank_avg_20260107_2057.csv
- outputs/self_training_runs/run_20260108_180837/submissions/submission.csv
- outputs/miracle_v2_runs/run_20260108_180757/submissions/best_weighted_all.csv
- outputs/blend_runs/run_20260108_201645/submissions/best2_rank.csv (optional)

Outputs:
- outputs/submissions/sub_final_*.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ANTIBIOTICS = [
    "Ampicillin",
    "Amoxicillin_Clavulanic_acid",
    "Cefotaxime",
    "Cefuroxime",
    "Ciprofloxacin",
    "Ertapenem",
    "Imipenem",
    "Levofloxacin",
]


@dataclass(frozen=True)
class Inputs:
    mega: Path
    self_train: Path
    miracle: Path
    best2_rank: Path | None


def _read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = ["sample_id", *ANTIBIOTICS]
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing or extra:
        raise ValueError(
            f"Unexpected columns in {path}: missing={missing}, extra={extra}, got={list(df.columns)}"
        )
    # Reorder for consistency (some saved submissions have different column order).
    return df[expected_cols]


def _assert_alignment(dfs: list[pd.DataFrame], paths: list[Path]) -> None:
    base = dfs[0]["sample_id"].values
    for df, path in zip(dfs[1:], paths[1:]):
        if not np.array_equal(base, df["sample_id"].values):
            raise ValueError(f"sample_id mismatch vs first input: {path}")


def _avg(dfs: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    if weights is None:
        weights = [1.0 / len(dfs)] * len(dfs)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()

    out = dfs[0][["sample_id"]].copy()
    for ab in ANTIBIOTICS:
        vals = np.stack([df[ab].values for df in dfs], axis=0)
        out[ab] = (vals.T @ weights_arr).astype(np.float64)
    return out


def _rank_avg(dfs: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    if weights is None:
        weights = [1.0 / len(dfs)] * len(dfs)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()

    out = dfs[0][["sample_id"]].copy()
    for ab in ANTIBIOTICS:
        ranks = []
        for df in dfs:
            ranks.append(df[ab].rank(pct=True, method="average").values)
        ranks = np.stack(ranks, axis=0)  # (n_models, n)
        out[ab] = (ranks.T @ weights_arr).astype(np.float64)
    return out


def _write(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> int:
    inputs = Inputs(
        mega=Path("outputs/submissions/sub_mega_blend_rank_avg_20260107_2057.csv"),
        self_train=Path("outputs/self_training_runs/run_20260108_180837/submissions/submission.csv"),
        miracle=Path("outputs/miracle_v2_runs/run_20260108_180757/submissions/best_weighted_all.csv"),
        best2_rank=Path("outputs/blend_runs/run_20260108_201645/submissions/best2_rank.csv"),
    )

    missing = [p for p in [inputs.mega, inputs.self_train, inputs.miracle] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    mega = _read_submission(inputs.mega)
    st = _read_submission(inputs.self_train)
    m2 = _read_submission(inputs.miracle)

    dfs = [mega, st, m2]
    paths = [inputs.mega, inputs.self_train, inputs.miracle]

    best2 = None
    if inputs.best2_rank is not None and inputs.best2_rank.exists():
        best2 = _read_submission(inputs.best2_rank)
        dfs_best2 = [mega, best2]
        paths_best2 = [inputs.mega, inputs.best2_rank]
        _assert_alignment(dfs_best2, paths_best2)

    _assert_alignment(dfs, paths)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/submissions")

    outputs: dict[str, pd.DataFrame] = {}

    # 2-way Mega + ST (often best: mega robustness + ST shift-awareness)
    outputs["mega_st_rank"] = _rank_avg([mega, st])
    outputs["mega_st_avg"] = _avg([mega, st])
    outputs["mega_st_avg_70_30"] = _avg([mega, st], weights=[0.7, 0.3])
    outputs["mega_st_avg_60_40"] = _avg([mega, st], weights=[0.6, 0.4])

    # 3-way Mega + ST + Miracle (diversity + shift-awareness)
    outputs["mega_st_m2_rank"] = _rank_avg([mega, st, m2])
    outputs["mega_st_m2_avg"] = _avg([mega, st, m2])
    outputs["mega_st_m2_avg_50_25_25"] = _avg([mega, st, m2], weights=[0.5, 0.25, 0.25])

    # Mega + (ST+Miracle already rank-averaged by earlier tooling)
    if best2 is not None:
        outputs["mega_best2rank_rank"] = _rank_avg([mega, best2])
        outputs["mega_best2rank_avg"] = _avg([mega, best2])

    written: list[Path] = []
    for name, df in outputs.items():
        out_path = out_dir / f"sub_final_{name}_{ts}.csv"
        _write(df, out_path)
        written.append(out_path)

    # Also copy the exact originals we blended (for convenience / re-submission)
    for tag, src in [
        ("mega_original", inputs.mega),
        ("self_train_original", inputs.self_train),
        ("miracle_original", inputs.miracle),
    ]:
        df = _read_submission(src)
        out_path = out_dir / f"sub_final_{tag}_{ts}.csv"
        _write(df, out_path)
        written.append(out_path)

    # Write a small manifest so you can see what was produced.
    manifest = out_dir / f"sub_final_manifest_{ts}.txt"
    manifest.write_text("\n".join(str(p) for p in written) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
