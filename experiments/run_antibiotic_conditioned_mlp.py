#!/usr/bin/env python3
"""Antibiotic-conditioned multi-task MLP (single model for all antibiotics).

Idea
- Train one big binary classifier over (sample, antibiotic) pairs.
- Inputs: MALDI features (optionally PCA-reduced) + species + antibiotic identity.
- Handles partial labels by masking: only trains on available labels.
- Uses a *proper* validation set sampled from fully-labeled rows to match the test
  species distribution (critical for this competition).

Outputs
- Writes run artifacts and a Kaggle submission CSV to outputs/antibiotic_mlp_runs/.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
OUTPUT_BASE = Path("/home/centcom/data/ml_kaggle/outputs")

# Constants
ANTIBIOTICS = [
    "Ampicillin",
    "Levofloxacin",
    "Ciprofloxacin",
    "Imipenem",
    "Amoxicillin_Clavulanic_acid",
    "Ertapenem",
    "Cefotaxime",
    "Cefuroxime",
]

ANTIBIOTIC_SHORT = {
    "Ampicillin": "AMP",
    "Amoxicillin_Clavulanic_acid": "AMC",
    "Cefotaxime": "CTX",
    "Cefuroxime": "CXM",
    "Ciprofloxacin": "CIP",
    "Ertapenem": "ETP",
    "Imipenem": "IPM",
    "Levofloxacin": "LVX",
}

SPECIES_NAMES = {0: "E.coli", 1: "K.pneumoniae", 2: "P.mirabilis", 3: "P.aeruginosa"}

# Target test species distribution (from EDA)
TEST_SPECIES_DISTRIBUTION = {0: 0.269, 1: 0.508, 2: 0.193, 3: 0.030}

# Reweight training to match test distribution
SPECIES_WEIGHTS = {0: 1.5, 1: 2.0, 2: 1.5, 3: 0.1}

# Intrinsic resistance rules
INTRINSIC_RESISTANCE = {
    3: ["Ampicillin", "Amoxicillin_Clavulanic_acid", "Ertapenem", "Cefotaxime", "Cefuroxime"],
    2: ["Imipenem"],
}


@dataclass
class Config:
    pca_components: int = 512
    transductive_pca: bool = True
    batch_size: int = 1024
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    hidden: str = "1024,512,256"
    antibiotic_emb_dim: int = 16
    species_emb_dim: int = 8
    val_fraction: float = 0.2
    patience: int = 6
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_intrinsic_rules(preds: np.ndarray, species: np.ndarray) -> np.ndarray:
    preds = preds.copy()
    ab_to_idx = {ab: i for i, ab in enumerate(ANTIBIOTICS)}
    for sp_id, abs_list in INTRINSIC_RESISTANCE.items():
        sp_mask = species == sp_id
        for ab in abs_list:
            preds[sp_mask, ab_to_idx[ab]] = 1.0
    return preds


def load_raw() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    test_df = pd.read_csv(RAW_DIR / "test.csv")

    feat_cols = [c for c in train_df.columns if c.startswith("maldi_feature_")]
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[ANTIBIOTICS].values.astype(np.float32)
    species_train = train_df["species_id"].values.astype(np.int64)

    X_test = test_df[feat_cols].values.astype(np.float32)
    species_test = test_df["species_id"].values.astype(np.int64)
    sample_ids = test_df["sample_id"].values

    return X_train, y_train, species_train, X_test, species_test, sample_ids


def make_val_split_from_fully_labeled(
    y: np.ndarray,
    species: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). Val is sampled from fully-labeled rows only."""
    rng = np.random.default_rng(seed)
    fully = np.all(~np.isnan(y), axis=1)
    fully_idx = np.where(fully)[0]
    sp_full = species[fully_idx]

    n_val = int(len(fully_idx) * val_fraction)

    val_local = []
    train_local = []
    for sp_id in range(4):
        sp_mask = sp_full == sp_id
        sp_indices = np.where(sp_mask)[0]
        target = int(n_val * TEST_SPECIES_DISTRIBUTION[sp_id])
        target = min(target, len(sp_indices))
        rng.shuffle(sp_indices)
        val_local.extend(sp_indices[:target])
        train_local.extend(sp_indices[target:])

    val_local = np.array(val_local, dtype=int)
    train_local = np.array(train_local, dtype=int)
    rng.shuffle(val_local)
    rng.shuffle(train_local)

    val_idx = fully_idx[val_local]
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

    return train_idx, val_idx


def standardize_and_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    transductive: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    # Remove constant features based on TRAIN only (safe) to avoid degenerate scaling
    variances = X_train.var(axis=0)
    mask = variances > 1e-8

    X_train = X_train[:, mask]
    X_val = X_val[:, mask]
    X_test = X_test[:, mask]

    scaler = StandardScaler()

    # Optionally skip PCA entirely (often better for trees; sometimes better for MLP too)
    if n_components <= 0:
        if transductive:
            X_all = np.vstack([X_train, X_val, X_test])
            X_all = scaler.fit_transform(X_all)
            n_train = len(X_train)
            n_val = len(X_val)
            X_train_s = X_all[:n_train]
            X_val_s = X_all[n_train : n_train + n_val]
            X_test_s = X_all[n_train + n_val :]
        else:
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

        stats = {
            "kept_features": int(mask.sum()),
            "removed_constant": int((~mask).sum()),
            "pca_components": 0,
            "explained_variance": 0.0,
        }
        return X_train_s.astype(np.float32), X_val_s.astype(np.float32), X_test_s.astype(np.float32), stats

    if transductive:
        X_all = np.vstack([X_train, X_val, X_test])
        X_all = scaler.fit_transform(X_all)
        n_train = len(X_train)
        n_val = len(X_val)
        n_comp = min(n_components, X_all.shape[1], X_all.shape[0] - 1)
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
        X_all_pca = pca.fit_transform(X_all)
        X_train_pca = X_all_pca[:n_train]
        X_val_pca = X_all_pca[n_train : n_train + n_val]
        X_test_pca = X_all_pca[n_train + n_val :]
        evr = float(pca.explained_variance_ratio_.sum())
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        n_comp = min(n_components, X_train_s.shape[1], X_train_s.shape[0] - 1)
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
        X_train_pca = pca.fit_transform(X_train_s)
        X_val_pca = pca.transform(X_val_s)
        X_test_pca = pca.transform(X_test_s)
        evr = float(pca.explained_variance_ratio_.sum())

    stats = {
        "kept_features": int(mask.sum()),
        "removed_constant": int((~mask).sum()),
        "pca_components": int(X_train_pca.shape[1]),
        "explained_variance": evr,
    }

    return X_train_pca.astype(np.float32), X_val_pca.astype(np.float32), X_test_pca.astype(np.float32), stats


class PairDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        species: np.ndarray,
        ab_to_idx: Dict[str, int],
        pos_weight_by_ab: np.ndarray,
        species_weights: Dict[int, float],
    ):
        # Build (row, antibiotic) pairs for available labels only
        rows, abs_idx = np.where(~np.isnan(y))
        self.x = torch.from_numpy(X[rows])
        self.species = torch.from_numpy(species[rows].astype(np.int64))
        self.ab = torch.from_numpy(abs_idx.astype(np.int64))
        labels = y[rows, abs_idx].astype(np.float32)
        self.y = torch.from_numpy(labels)

        # Per-example weights: species shift + imbalance correction
        sp_w = np.array([species_weights.get(int(s), 1.0) for s in species[rows]], dtype=np.float32)
        pw = pos_weight_by_ab[abs_idx].astype(np.float32)
        w = sp_w * np.where(labels > 0.5, pw, 1.0)
        self.w = torch.from_numpy(w)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.species[idx], self.ab[idx], self.y[idx], self.w[idx]


class AntibioticConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_antibiotics: int = 8,
        n_species: int = 4,
        antibiotic_emb_dim: int = 16,
        species_emb_dim: int = 8,
        hidden: Tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.ab_emb = nn.Embedding(n_antibiotics, antibiotic_emb_dim)
        self.sp_emb = nn.Embedding(n_species, species_emb_dim)

        d0 = input_dim + antibiotic_emb_dim + species_emb_dim
        layers = []
        d_in = d0
        for h in hidden:
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, species: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        ab_e = self.ab_emb(ab)
        sp_e = self.sp_emb(species)
        z = torch.cat([x, ab_e, sp_e], dim=1)
        return self.net(z).squeeze(1)


@torch.no_grad()
def predict_matrix(
    model: nn.Module,
    X: np.ndarray,
    species: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    X_t = torch.from_numpy(X).to(device)
    sp_t = torch.from_numpy(species.astype(np.int64)).to(device)

    n = len(X)
    out = np.zeros((n, len(ANTIBIOTICS)), dtype=np.float32)

    for ab_idx in range(len(ANTIBIOTICS)):
        ab_t = torch.full((n,), ab_idx, dtype=torch.int64, device=device)
        preds = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            logits = model(X_t[start:end], sp_t[start:end], ab_t[start:end])
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        out[:, ab_idx] = np.concatenate(preds, axis=0)

    return out


def compute_mean_auc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    per_ab = {}
    aucs = []
    for j, ab in enumerate(ANTIBIOTICS):
        mask = ~np.isnan(y_true[:, j])
        if mask.sum() > 10 and len(np.unique(y_true[mask, j])) > 1:
            auc = roc_auc_score(y_true[mask, j], y_pred[mask, j])
            per_ab[ab] = float(auc)
            aucs.append(auc)
        else:
            per_ab[ab] = float("nan")
    return float(np.mean(aucs) if aucs else 0.0), per_ab


def parse_hidden(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", type=int, default=512, help="PCA components; set 0 to disable PCA")
    parser.add_argument("--no-transductive-pca", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden", type=str, default="1024,512,256")
    parser.add_argument("--ab-emb", type=int, default=16)
    parser.add_argument("--sp-emb", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = Config(
        pca_components=args.pca,
        transductive_pca=not args.no_transductive_pca,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden=args.hidden,
        antibiotic_emb_dim=args.ab_emb,
        species_emb_dim=args.sp_emb,
        val_fraction=args.val_frac,
        patience=args.patience,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    # Output dir
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / "antibiotic_mlp_runs" / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_full, y_full, species_full, X_test_raw, species_test, sample_ids = load_raw()

    train_idx, val_idx = make_val_split_from_fully_labeled(y_full, species_full, cfg.val_fraction, cfg.seed)
    X_train_raw, y_train = X_full[train_idx], y_full[train_idx]
    sp_train = species_full[train_idx]
    X_val_raw, y_val = X_full[val_idx], y_full[val_idx]
    sp_val = species_full[val_idx]

    # PCA
    X_train, X_val, X_test, pca_stats = standardize_and_pca(
        X_train_raw,
        X_val_raw,
        X_test_raw,
        n_components=cfg.pca_components,
        transductive=cfg.transductive_pca,
        seed=cfg.seed,
    )

    # Pos weights per antibiotic from training labels
    pos_weight_by_ab = np.ones(len(ANTIBIOTICS), dtype=np.float32)
    for j in range(len(ANTIBIOTICS)):
        col = y_train[:, j]
        m = ~np.isnan(col)
        if m.sum() > 20:
            pos = float(np.sum(col[m] > 0.5))
            neg = float(np.sum(col[m] <= 0.5))
            if pos > 0:
                pos_weight_by_ab[j] = np.clip(neg / pos, 0.5, 20.0)

    # Dataset
    ab_to_idx = {ab: i for i, ab in enumerate(ANTIBIOTICS)}
    ds_train = PairDataset(X_train, y_train, sp_train, ab_to_idx, pos_weight_by_ab, SPECIES_WEIGHTS)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    hidden = parse_hidden(cfg.hidden)
    model = AntibioticConditionedMLP(
        input_dim=X_train.shape[1],
        antibiotic_emb_dim=cfg.antibiotic_emb_dim,
        species_emb_dim=cfg.species_emb_dim,
        hidden=hidden,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_mean_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for xb, spb, abb, yb, wb in dl_train:
            xb = xb.to(device, non_blocking=True)
            spb = spb.to(device, non_blocking=True)
            abb = abb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb, spb, abb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += float(loss.detach().cpu().item()) * len(yb)
            n_seen += len(yb)

        train_loss = running / max(1, n_seen)

        # Validation
        val_preds = predict_matrix(model, X_val, sp_val, device=device, batch_size=cfg.batch_size)
        val_preds = apply_intrinsic_rules(val_preds, sp_val)
        mean_auc, per_ab = compute_mean_auc(y_val, val_preds)

        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | val_mean_auc={mean_auc:.4f}"
        )

        if mean_auc > best_mean_auc + 1e-4:
            best_mean_auc = mean_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping (patience={cfg.patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final predictions
    val_preds = predict_matrix(model, X_val, sp_val, device=device, batch_size=cfg.batch_size)
    val_preds = apply_intrinsic_rules(val_preds, sp_val)
    final_mean_auc, final_per_ab = compute_mean_auc(y_val, val_preds)

    test_preds = predict_matrix(model, X_test, species_test, device=device, batch_size=cfg.batch_size)
    test_preds = apply_intrinsic_rules(test_preds, species_test)

    # Save submission
    submissions_dir = run_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    sub = pd.DataFrame({"sample_id": sample_ids, **{ab: test_preds[:, j] for j, ab in enumerate(ANTIBIOTICS)}})
    sub_path = submissions_dir / "submission.csv"
    sub.to_csv(sub_path, index=False)

    # Save results
    results = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "val_mean_auc": float(final_mean_auc),
        "val_per_antibiotic": final_per_ab,
        "config": asdict(cfg),
        "pca": pca_stats,
        "n_train_pairs": int(len(ds_train)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), run_dir / "model.pt")

    print("\nSaved:")
    print("-", sub_path)
    print("-", run_dir / "results.json")
    print(f"Val Mean AUC: {final_mean_auc:.4f}")


if __name__ == "__main__":
    main()
