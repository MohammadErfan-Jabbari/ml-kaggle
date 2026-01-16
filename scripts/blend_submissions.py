#!/usr/bin/env python
"""Blend multiple Kaggle submission CSVs.

Default is per-antibiotic rank averaging (works well when models are on different calibration scales).

Example:
  uv run python scripts/blend_submissions.py \
    --out outputs/submissions/blend_rankavg_mega_selftrain.csv \
    --input outputs/experiments/mega_blend_rank_avg_20260108_2305.csv \
    --input outputs/self_training_runs/run_20260108_193910/submissions/submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def read_and_align(path: Path, base_ids: pd.Series, antibiotic_cols: list[str]) -> np.ndarray:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id in {path}")
    missing = [c for c in antibiotic_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    merged = pd.DataFrame({"sample_id": base_ids}).merge(
        df[["sample_id", *antibiotic_cols]], on="sample_id", how="left"
    )
    if merged[antibiotic_cols].isna().any().any():
        raise ValueError(f"{path} does not cover all sample_id values")
    preds = merged[antibiotic_cols].to_numpy(dtype=float)
    return np.clip(preds, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="Submission CSV path (repeatable)")
    ap.add_argument("--out", required=True, help="Output blended submission path")
    ap.add_argument(
        "--method",
        choices=["rankavg", "mean"],
        default="rankavg",
        help="Blend method: rankavg (default) or mean",
    )
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.input]
    if len(input_paths) < 2:
        raise SystemExit("Need at least 2 --input files")

    base = pd.read_csv(input_paths[0])
    if "sample_id" not in base.columns:
        raise SystemExit(f"Missing sample_id in {input_paths[0]}")

    antibiotic_cols = [c for c in base.columns if c != "sample_id"]
    if not antibiotic_cols:
        raise SystemExit(f"No antibiotic columns found in {input_paths[0]}")

    base_ids = base["sample_id"].copy()

    preds_list = [read_and_align(p, base_ids, antibiotic_cols) for p in input_paths]

    if args.method == "mean":
        blended = np.mean(preds_list, axis=0)
    else:
        # Per-antibiotic rank averaging
        rank_sum = np.zeros_like(preds_list[0])
        for preds in preds_list:
            for j in range(preds.shape[1]):
                rank_sum[:, j] += rankdata(preds[:, j]) / len(preds)
        blended = rank_sum / len(preds_list)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"sample_id": base_ids, **{c: blended[:, i] for i, c in enumerate(antibiotic_cols)}})
    out_df.to_csv(out_path, index=False)
    print(f"Wrote blended submission: {out_path}")


if __name__ == "__main__":
    main()
