"""
Backfill wrist depth and behind-back features into an existing 26-feature
dataset CSV, producing a 31-feature `<input>_z.csv` sibling.

Adds five new columns:
    Rel_LeftWrist_Z   Rel_RightWrist_Z   (from MediaPipe Pose)
    Left_Wrist_Behind   Right_Wrist_Behind   Hands_Behind_Back_Count
                                         (derived from the Z columns)

Everything else in the CSV (labels, smoothing, ball coords, X/Y/Vis for all
landmarks) is preserved untouched. For each unique Video_ID we re-open the
source MP4 and run MediaPipe Pose to grab `z` for the two wrists; the X/Y
already in the CSV are NOT recomputed.

Run:
    python 15_backfill_wrist_z.py --in ../data/bskt/current_datasets/training_dataset_smooth_tracker.csv
    python 15_backfill_wrist_z.py --in ../data/bskt/current_datasets/validation_dataset_smooth_tracker.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


LM = mp.solutions.pose.PoseLandmark
I_LWRIST = LM.LEFT_WRIST.value
I_RWRIST = LM.RIGHT_WRIST.value

# Threshold matches inject_derived_features in 10_build_dribble_dataset.py.
BEHIND_THRESHOLD = 0.05

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill-z")


def extract_wrist_z_for_video(video_path: Path) -> dict[int, tuple[float, float]]:
    """Run MediaPipe Pose on every frame of `video_path` and return
    {Frame_ID: (left_wrist_z, right_wrist_z)}.

    Frames where pose tracking failed are absent from the dict (the caller
    leaves them as NaN). MediaPipe settings match 10_build_dribble_dataset.py
    so the depth values come from the same model the dataset was built with.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Could not open %s", video_path)
        return {}

    result: dict[int, tuple[float, float]] = {}
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks is not None:
                lms = res.pose_landmarks.landmark
                # MediaPipe's z is already hip-relative (negative in front of
                # the camera, positive behind), so we store it as-is.
                result[frame_id] = (float(lms[I_LWRIST].z), float(lms[I_RWRIST].z))
            frame_id += 1
    finally:
        pose.close()
        cap.release()

    log.info("  %s: %d / %d frames had a valid pose", video_path.name, len(result), frame_id)
    return result


def insert_or_assign(df: pd.DataFrame, col_name: str, values, after: Optional[str]) -> None:
    """Insert `col_name` into `df` immediately after `after` (or at the end
    if `after` is None or not found). If the column already exists, overwrite
    values in place without changing position — makes the script idempotent
    if you re-run on a CSV that's already been backfilled."""
    if col_name in df.columns:
        df[col_name] = values
        return
    if after and after in df.columns:
        loc = df.columns.get_loc(after) + 1
    else:
        loc = len(df.columns)
    df.insert(loc, col_name, values)


def backfill(input_csv: Path, videos_dir: Path, output_csv: Path) -> None:
    log.info("Reading %s", input_csv)
    df = pd.read_csv(input_csv)
    log.info("  %d rows, %d columns", len(df), len(df.columns))

    if "Video_ID" not in df.columns or "Frame_ID" not in df.columns:
        raise SystemExit("Input CSV must have Video_ID and Frame_ID columns.")

    left_z  = np.full(len(df), np.nan, dtype=np.float32)
    right_z = np.full(len(df), np.nan, dtype=np.float32)

    missing_videos: list[str] = []
    for video_id, group in df.groupby("Video_ID", sort=False):
        video_path = videos_dir / str(video_id)
        if not video_path.exists():
            log.warning("Missing video %s — leaving Z as NaN for %d rows.",
                        video_path, len(group))
            missing_videos.append(str(video_id))
            continue

        log.info("Processing %s (%d CSV rows)", video_id, len(group))
        z_by_frame = extract_wrist_z_for_video(video_path)

        for csv_idx, frame_id in zip(group.index, group["Frame_ID"].astype(int)):
            entry = z_by_frame.get(int(frame_id))
            if entry is not None:
                left_z[csv_idx], right_z[csv_idx] = entry

    # Place the Z columns next to the matching wrist X/Y/Vis block. Fall back
    # to end-of-frame if the anchor isn't present (custom-shaped CSV).
    insert_or_assign(df, "Rel_LeftWrist_Z",  left_z,  after="Rel_LeftWrist_Y")
    insert_or_assign(df, "Rel_RightWrist_Z", right_z, after="Rel_RightWrist_Y")

    # Discrete behind-back indicators. NaN > 0.05 evaluates to False, so
    # missing-pose frames cleanly get 0.0 — same semantics as
    # inject_derived_features in 10_build_dribble_dataset.py.
    left_behind  = (df["Rel_LeftWrist_Z"]  > BEHIND_THRESHOLD).astype(float)
    right_behind = (df["Rel_RightWrist_Z"] > BEHIND_THRESHOLD).astype(float)
    behind_count = left_behind + right_behind

    # Place these alongside the other derived features (Delta_Ball_X is the
    # last per-frame derived column in the canonical layout, sitting just
    # before the labels).
    anchor = "Delta_Ball_X" if "Delta_Ball_X" in df.columns else None
    insert_or_assign(df, "Left_Wrist_Behind",       left_behind,  after=anchor)
    insert_or_assign(df, "Right_Wrist_Behind",      right_behind, after="Left_Wrist_Behind")
    insert_or_assign(df, "Hands_Behind_Back_Count", behind_count, after="Right_Wrist_Behind")

    df.to_csv(output_csv, index=False)
    log.info("Wrote %d rows -> %s", len(df), output_csv)

    # Sanity report — what fraction of rows actually got Z values, and what
    # the behind-back distribution looks like. Catches "videos missing" or
    # "MediaPipe failed on this player" silently producing all-zero features.
    z_filled = int(df["Rel_LeftWrist_Z"].notna().sum())
    pct = 100.0 * z_filled / max(1, len(df))
    log.info("Rel_LeftWrist_Z populated for %d / %d rows (%.1f%%).",
             z_filled, len(df), pct)
    counts = df["Hands_Behind_Back_Count"].value_counts(dropna=False).sort_index().to_dict()
    log.info("Hands_Behind_Back_Count distribution: %s", counts)
    if missing_videos:
        log.warning("Skipped %d missing video(s): %s",
                    len(missing_videos), missing_videos[:5] + (["..."] if len(missing_videos) > 5 else []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True,
                    help="Input CSV (the existing 26-feature dataset).")
    ap.add_argument("--videos", default="../raw_dribbling_videos",
                    help="Directory containing the source MP4s referenced by Video_ID.")
    ap.add_argument("--out", default=None,
                    help="Output CSV path. Default: <input>_z.csv next to the input.")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    videos_dir = Path(args.videos)
    if not in_path.is_file():
        raise SystemExit(f"Input CSV not found: {in_path}")
    if not videos_dir.is_dir():
        raise SystemExit(f"Videos directory not found: {videos_dir}")

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_z.csv")
    backfill(in_path, videos_dir, out_path)


if __name__ == "__main__":
    main()
