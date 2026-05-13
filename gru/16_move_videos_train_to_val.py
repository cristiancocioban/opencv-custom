"""
Move rows for specified Video_IDs from a training CSV into a validation CSV.

Use case: rebalance an existing train/val split (e.g., to grow the val set
without re-running the dataset pipeline). Both CSVs must share the same
column schema; matching rows are removed from training and appended to
validation. Originals are backed up to `*.bak` and overwritten in place.

Run:
    # Defaults point at the *_z.csv pair used by 60_build_gru_model.py.
    python 16_move_videos_train_to_val.py --videos crossover_07_05_01.mp4 crossover_02_05_7.mp4

    # Custom CSV paths:
    python 16_move_videos_train_to_val.py \\
        --train ../data/bskt/current_datasets/training_dataset_smooth_tracker_z.csv \\
        --val   ../data/bskt/current_datasets/validation_dataset_smooth_tracker_z.csv \\
        --videos crossover_07_05_01.mp4

    # Preview what would happen without writing anything:
    python 16_move_videos_train_to_val.py --videos crossover_07_05_01.mp4 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("move-videos")


def move_videos(train_csv: Path, val_csv: Path, video_ids: list[str], dry_run: bool) -> None:
    log.info("Reading %s", train_csv)
    train_df = pd.read_csv(train_csv)
    log.info("  %d rows, %d columns", len(train_df), len(train_df.columns))

    log.info("Reading %s", val_csv)
    val_df = pd.read_csv(val_csv)
    log.info("  %d rows, %d columns", len(val_df), len(val_df.columns))

    # Refuse to merge mismatched schemas — silently NaN-filling missing
    # columns would corrupt training in non-obvious ways.
    if list(train_df.columns) != list(val_df.columns):
        train_only = set(train_df.columns) - set(val_df.columns)
        val_only = set(val_df.columns) - set(train_df.columns)
        raise SystemExit(
            "Column mismatch between training and validation CSVs:\n"
            f"  In training only:    {sorted(train_only)}\n"
            f"  In validation only:  {sorted(val_only)}\n"
            "Re-run the backfill on whichever CSV is missing the new columns first."
        )

    requested = list(dict.fromkeys(video_ids))  # preserve order, drop dupes
    train_video_set = set(train_df["Video_ID"].unique())
    val_video_set = set(val_df["Video_ID"].unique())

    not_in_train   = [v for v in requested if v not in train_video_set]
    already_in_val = [v for v in requested if v in val_video_set]

    if not_in_train:
        log.warning("Video_ID(s) not found in training set, skipping: %s", not_in_train)
    if already_in_val:
        log.warning("Video_ID(s) ALREADY present in validation set — moving them "
                    "from training will create duplicate rows in val: %s", already_in_val)

    to_move = [v for v in requested if v in train_video_set]
    if not to_move:
        raise SystemExit("No requested Video_ID(s) found in the training CSV. Nothing to do.")

    move_mask = train_df["Video_ID"].isin(to_move)
    moving_rows  = train_df[move_mask]
    keeping_rows = train_df[~move_mask]

    log.info("Moving %d Video_ID(s):", len(to_move))
    for vid in to_move:
        n = int((moving_rows["Video_ID"] == vid).sum())
        log.info("  %s -> %d rows", vid, n)

    new_val_df   = pd.concat([val_df, moving_rows], ignore_index=True)
    new_train_df = keeping_rows.reset_index(drop=True)

    log.info("Result: train %d -> %d rows; val %d -> %d rows",
             len(train_df), len(new_train_df), len(val_df), len(new_val_df))

    if dry_run:
        log.info("--dry-run set; not writing any files.")
        return

    # Cheap safety net before overwriting in place.
    train_backup = train_csv.with_suffix(train_csv.suffix + ".bak")
    val_backup   = val_csv.with_suffix(val_csv.suffix + ".bak")
    shutil.copy2(train_csv, train_backup)
    shutil.copy2(val_csv,   val_backup)
    log.info("Backed up originals -> %s, %s", train_backup, val_backup)

    new_train_df.to_csv(train_csv, index=False)
    new_val_df.to_csv(val_csv, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows).",
             train_csv, len(new_train_df), val_csv, len(new_val_df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/bskt/current_datasets/training_dataset_smooth_tracker_z.csv",
                    help="Path to the training CSV (overwritten in place).")
    ap.add_argument("--val",   default="../data/bskt/current_datasets/validation_dataset_smooth_tracker_z.csv",
                    help="Path to the validation CSV (overwritten in place).")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="Video_ID values to move (use the exact filename, "
                         "e.g. crossover_07_05_01.mp4).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen without modifying any files.")
    args = ap.parse_args()

    train_csv = Path(args.train)
    val_csv   = Path(args.val)
    if not train_csv.is_file():
        raise SystemExit(f"Training CSV not found: {train_csv}")
    if not val_csv.is_file():
        raise SystemExit(f"Validation CSV not found: {val_csv}")

    move_videos(train_csv, val_csv, args.videos, args.dry_run)


if __name__ == "__main__":
    main()
