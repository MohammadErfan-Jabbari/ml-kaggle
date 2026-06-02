#!/usr/bin/env python3
"""
CLI script to update Kaggle leaderboard scores for submissions.

Usage:
    # Update specific submission
    python scripts/update_leaderboard.py --sub-id 1 --public-lb 0.8567

    # Update with both public and private
    python scripts/update_leaderboard.py --sub-id 1 --public-lb 0.8567 --private-lb 0.8432

    # Interactive mode (prompts for latest submission)
    python scripts/update_leaderboard.py
"""

import argparse
import sys

# Add src to path
sys.path.insert(0, "/home/centcom/data/ml_kaggle")

from src.utils.submission_tracker import (
    update_leaderboard_score,
    get_submission_history,
    print_submission_history
)


def main():
    parser = argparse.ArgumentParser(description="Update Kaggle leaderboard scores")
    parser.add_argument("--sub-id", type=int, help="Submission ID to update")
    parser.add_argument("--public-lb", type=float, help="Public leaderboard score")
    parser.add_argument("--private-lb", type=float, help="Private leaderboard score")

    args = parser.parse_args()

    # Show current history
    print_submission_history()

    # Interactive mode if no arguments provided
    if args.sub_id is None:
        history = get_submission_history()
        if not history["submissions"]:
            print("\nNo submissions found. Create a submission first.")
            return

        # Find most recent submission without public LB
        latest_pending = None
        for sub in reversed(history["submissions"]):
            if sub["public_lb"] is None:
                latest_pending = sub
                break

        if latest_pending:
            print(f"\nMost recent pending submission: #{latest_pending['sub_id']} ({latest_pending['filename']})")
            try:
                sub_id = latest_pending["sub_id"]
                public_lb = input(f"Enter Public LB score for submission #{sub_id} (or press Enter to skip): ").strip()
                private_lb = input(f"Enter Private LB score for submission #{sub_id} (or press Enter to skip): ").strip()

                args.sub_id = sub_id
                args.public_lb = float(public_lb) if public_lb else None
                args.private_lb = float(private_lb) if private_lb else None
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.")
                return
        else:
            print("\nNo pending submissions found (all have public LB scores).")
            return

    # Validate inputs
    if args.sub_id is None:
        print("\nError: --sub-id is required in non-interactive mode")
        return

    if args.public_lb is None and args.private_lb is None:
        print("\nError: At least one of --public-lb or --private-lb must be provided")
        return

    # Update the score
    success = update_leaderboard_score(args.sub_id, args.public_lb, args.private_lb)

    if success:
        print(f"\n✅ Updated submission #{args.sub_id}")
        if args.public_lb is not None:
            print(f"   Public LB: {args.public_lb:.4f}")
        if args.private_lb is not None:
            print(f"   Private LB: {args.private_lb:.4f}")
    else:
        print(f"\n❌ Failed to update submission #{args.sub_id} (not found)")


if __name__ == "__main__":
    main()
