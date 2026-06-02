"""Submission tracking utilities for Kaggle competition."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# Paths
TRACKER_DIR = Path("/home/centcom/data/ml_kaggle/knowledge/submissions")
TRACKER_CSV = TRACKER_DIR / "submissions_tracker.csv"
TRACKER_LOG = TRACKER_DIR / "submissions_log.md"


def generate_submission_name(model_type: str, version_desc: str) -> str:
    """
    Generate a unique submission filename.

    Pattern: sub_{model}_{version}_{timestamp}.csv

    Args:
        model_type: Abbreviated model type (lgb, mlp, cnn, ensemble, etc.)
        version_desc: Descriptive suffix (baseline, v2, intrinsic_rules, etc.)

    Returns:
        Filename string
    """
    # Clean version_desc (replace spaces with underscores, limit length)
    version_clean = version_desc.replace(" ", "_").replace("-", "_")[:30]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"sub_{model_type}_{version_clean}_{timestamp}.csv"


def _init_tracker_csv():
    """Initialize the tracker CSV file with headers if it doesn't exist."""
    if not TRACKER_CSV.exists():
        TRACKER_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACKER_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sub_id", "filename", "timestamp", "model_type", "version_desc",
                "oof_mean_auc", "oof_per_antibiotic", "public_lb", "private_lb", "notes"
            ])


def _init_tracker_log():
    """Initialize the tracker log markdown file if it doesn't exist."""
    if not TRACKER_LOG.exists():
        TRACKER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACKER_LOG, "w") as f:
            f.write("# Submission Log\n\n")
            f.write("This file tracks all Kaggle submissions with their local and leaderboard performance.\n\n")
            f.write("---\n\n")


def log_submission(
    filename: str,
    model_type: str,
    version_desc: str,
    metrics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    notes: str = ""
) -> int:
    """
    Log a new submission to both CSV and markdown log.

    Args:
        filename: Name of the submission file
        model_type: Type of model (lgb, mlp, cnn, ensemble, etc.)
        version_desc: Human-readable description
        metrics: Dictionary with 'mean_auc' and 'per_antibiotic' (dict of antibiotic -> AUC)
        config: Optional model configuration dictionary
        notes: Any additional notes

    Returns:
        The submission ID (auto-incremented)
    """
    # Initialize files if needed
    _init_tracker_csv()
    _init_tracker_log()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    oof_mean_auc = metrics.get("mean_auc", 0.0)
    per_antibiotic = metrics.get("per_antibiotic", {})

    # Get next sub_id
    sub_id = 1
    if TRACKER_CSV.exists() and TRACKER_CSV.stat().st_size > 0:
        with open(TRACKER_CSV, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                sub_id = int(rows[-1]["sub_id"]) + 1

    # Write to CSV
    with open(TRACKER_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            sub_id,
            filename,
            timestamp,
            model_type,
            version_desc,
            f"{oof_mean_auc:.4f}",
            json.dumps(per_antibiotic),
            "",  # public_lb (to be filled later)
            "",  # private_lb (to be filled later)
            notes
        ])

    # Write to markdown log
    with open(TRACKER_LOG, "a") as f:
        f.write(f"## Submission #{sub_id}: {version_desc.title()}\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**File**: `{filename}`\n")
        f.write(f"**Model**: {model_type.upper()}\n")
        f.write(f"**Description**: {version_desc}\n\n")

        # Local OOF Performance
        f.write("### Local OOF Performance\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean AUC | {oof_mean_auc:.4f} |\n")
        for antibiotic, auc in per_antibiotic.items():
            f.write(f"| {antibiotic} | {auc:.4f} |\n")
        f.write("\n")

        # Config details
        if config:
            f.write("### Configuration\n")
            f.write("```python\n")
            f.write(json.dumps(config, indent=2))
            f.write("\n```\n\n")

        # Kaggle Leaderboard
        f.write("### Kaggle Leaderboard\n")
        f.write("- **Public LB**: _pending submission_\n")
        f.write("- **Private LB**: _pending final results_\n\n")

        # Notes
        if notes:
            f.write(f"### Notes\n")
            f.write(f"{notes}\n\n")

        f.write("---\n\n")

    return sub_id


def update_leaderboard_score(
    sub_id: int,
    public_lb: Optional[float] = None,
    private_lb: Optional[float] = None
) -> bool:
    """
    Update leaderboard scores for a submission.

    Args:
        sub_id: Submission ID to update
        public_lb: Public leaderboard score
        private_lb: Private leaderboard score (final)

    Returns:
        True if updated successfully, False if submission not found
    """
    if not TRACKER_CSV.exists():
        return False

    # Read all rows
    with open(TRACKER_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find and update the row
    updated = False
    for row in rows:
        if int(row["sub_id"]) == sub_id:
            if public_lb is not None:
                row["public_lb"] = f"{public_lb:.4f}"
            if private_lb is not None:
                row["private_lb"] = f"{private_lb:.4f}"
            updated = True
            break

    if not updated:
        return False

    # Write back
    with open(TRACKER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Update markdown log
    _update_log_leaderboard(sub_id, public_lb, private_lb)

    return True


def _update_log_leaderboard(sub_id: int, public_lb: Optional[float], private_lb: Optional[float]):
    """Update the markdown log with leaderboard scores."""
    if not TRACKER_LOG.exists():
        return

    with open(TRACKER_LOG, "r") as f:
        content = f.read()

    # Find the submission section and update
    lines = content.split("\n")
    in_submission = False
    found_lb_section = False

    for i, line in enumerate(lines):
        if f"## Submission #{sub_id}:" in line:
            in_submission = True
        elif in_submission and "### Kaggle Leaderboard" in line:
            found_lb_section = True
        elif found_lb_section:
            if "**Public LB**" in line and public_lb is not None:
                lines[i] = f"- **Public LB**: {public_lb:.4f}\n"
            elif "**Private LB**" in line and private_lb is not None:
                lines[i] = f"- **Private LB**: {private_lb:.4f}\n"
                break  # Done with this submission

    with open(TRACKER_LOG, "w") as f:
        f.write("\n".join(lines))


def get_submission_history() -> Dict[str, Any]:
    """
    Get all submission history as a dictionary.

    Returns:
        Dictionary with 'submissions' list and 'best' submission info
    """
    if not TRACKER_CSV.exists() or TRACKER_CSV.stat().st_size == 0:
        return {"submissions": [], "best": None}

    with open(TRACKER_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    submissions = []
    best_public_lb = None
    best_private_lb = None

    for row in rows:
        sub = {
            "sub_id": int(row["sub_id"]),
            "filename": row["filename"],
            "timestamp": row["timestamp"],
            "model_type": row["model_type"],
            "version_desc": row["version_desc"],
            "oof_mean_auc": float(row["oof_mean_auc"]) if row["oof_mean_auc"] else None,
            "oof_per_antibiotic": json.loads(row["oof_per_antibiotic"]) if row["oof_per_antibiotic"] else {},
            "public_lb": float(row["public_lb"]) if row["public_lb"] else None,
            "private_lb": float(row["private_lb"]) if row["private_lb"] else None,
            "notes": row["notes"]
        }
        submissions.append(sub)

        # Track best scores
        if sub["public_lb"] is not None:
            if best_public_lb is None or sub["public_lb"] > best_public_lb["score"]:
                best_public_lb = {"sub_id": sub["sub_id"], "score": sub["public_lb"]}
        if sub["private_lb"] is not None:
            if best_private_lb is None or sub["private_lb"] > best_private_lb["score"]:
                best_private_lb = {"sub_id": sub["sub_id"], "score": sub["private_lb"]}

    return {
        "submissions": submissions,
        "best_public_lb": best_public_lb,
        "best_private_lb": best_private_lb
    }


def print_submission_history():
    """Print all submission history in a formatted table."""
    history = get_submission_history()

    if not history["submissions"]:
        print("No submissions tracked yet.")
        return

    print("\n" + "=" * 120)
    print("Submission History")
    print("=" * 120)

    for sub in history["submissions"]:
        print(f"\nSubmission #{sub['sub_id']}: {sub['version_desc']}")
        print(f"  File: {sub['filename']}")
        print(f"  Date: {sub['timestamp']}")
        print(f"  Model: {sub['model_type']}")
        print(f"  OOF Mean AUC: {sub['oof_mean_auc']:.4f}" if sub['oof_mean_auc'] else "  OOF Mean AUC: N/A")
        if sub['public_lb']:
            print(f"  Public LB: {sub['public_lb']:.4f}")
        if sub['private_lb']:
            print(f"  Private LB: {sub['private_lb']:.4f}")

    print("\n" + "=" * 120)

    if history["best_public_lb"]:
        print(f"Best Public LB: {history['best_public_lb']['score']:.4f} (Submission #{history['best_public_lb']['sub_id']})")
    if history["best_private_lb"]:
        print(f"Best Private LB: {history['best_private_lb']['score']:.4f} (Submission #{history['best_private_lb']['sub_id']})")


if __name__ == "__main__":
    # Test the module
    print_submission_history()
