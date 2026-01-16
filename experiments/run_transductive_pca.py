#!/usr/bin/env python3
"""
Transductive PCA + Self-Training for AMR Prediction.

Wrapper script that uses standard PCA for dimensionality reduction.
"""

import argparse
import sys
from pathlib import Path

# Add experiments to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transductive_base import TransductiveConfig, run_transductive_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Transductive PCA + Self-Training for AMR prediction"
    )

    # DR settings
    parser.add_argument(
        "--n-components", type=int, default=100,
        help="Number of PCA components (default: 100)"
    )

    # Self-training settings
    parser.add_argument(
        "--max-iter", type=int, default=5,
        help="Maximum self-training iterations (default: 5)"
    )
    parser.add_argument(
        "--conf-high", type=float, default=0.85,
        help="High confidence threshold for pseudo-labeling (default: 0.85)"
    )
    parser.add_argument(
        "--conf-low", type=float, default=0.15,
        help="Low confidence threshold for pseudo-labeling (default: 0.15)"
    )
    parser.add_argument(
        "--pseudo-weight", type=float, default=0.5,
        help="Weight for pseudo-labeled samples (default: 0.5)"
    )

    # Model ensemble
    parser.add_argument(
        "--no-mlp", action="store_true",
        help="Disable MLP in ensemble"
    )
    parser.add_argument(
        "--no-xgb", action="store_true",
        help="Disable XGBoost in ensemble"
    )
    parser.add_argument(
        "--no-catboost", action="store_true",
        help="Disable CatBoost in ensemble"
    )

    # Runtime
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run quick smoke test with minimal settings"
    )

    args = parser.parse_args()

    # Adjust for smoke test
    n_components = 10 if args.smoke_test else args.n_components
    max_iterations = 2 if args.smoke_test else args.max_iter

    config = TransductiveConfig(
        dr_method='pca',
        n_components=n_components,
        max_iterations=max_iterations,
        confidence_threshold_high=args.conf_high,
        confidence_threshold_low=args.conf_low,
        pseudo_label_weight=args.pseudo_weight,
        use_mlp=not args.no_mlp,
        use_xgb=not args.no_xgb,
        use_catboost=not args.no_catboost,
        smoke_test=args.smoke_test,
    )

    results = run_transductive_pipeline(config)

    return results


if __name__ == "__main__":
    main()
