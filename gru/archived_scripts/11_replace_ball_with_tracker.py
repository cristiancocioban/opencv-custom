"""
11_replace_ball_with_tracker.py

Re-detect ball positions in existing training / validation CSVs using
cv2.BallTracker (YOLO + TrackerNano), then recompute ball-derived features.
Pose features and labels (Dribble / Crossover / Hand_Touch) are kept as-is
from the original CSV.

Updated columns:
  - Rel_Ball_X, Rel_Ball_Y       (ball center minus hip center, normalized)
  - Ball_Detected                (1.0 if BallTracker found the ball, else 0.0)
  - Dist_Ball_L_Wrist            (recomputed from new ball + existing wrist)
  - Dist_Ball_R_Wrist
  - Delta_Ball_X, Delta_Ball_Y   (per-video, forward-filled across misses)

Run:
    python 11_replace_ball_with_tracker.py
    python 11_replace_ball_with_tracker.py --videos-dir ../raw_dribbling_videos/processed

The Ball_Detected column is added if it doesn't already exist.
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


# ----------------------------- Configuration --------------------------------

# Output column order. Must match 10_build_dribble_dataset.py so downstream
# tooling (smoothing, training) can consume both files interchangeably.
CSV_COLUMNS = [
    "Video_ID", "Frame_ID", "Norm_Torso_Height",
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y", "LeftElbow_Vis",
    "Rel_RightElbow_X", "Rel_RightElbow_Y", "RightElbow_Vis",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "LeftWrist_Vis",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "RightWrist_Vis",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y", "LeftAnkle_Vis",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y", "RightAnkle_Vis",
    "Dist_Ball_L_Wrist", "Dist_Ball_R_Wrist",
    "Delta_Ball_Y", "Delta_Ball_X",
    "Dribble", "Crossover", "Hand_Touch",
    "Ball_Detected",
]

LM = mp.solutions.pose.PoseLandmark
I_LHIP = LM.LEFT_HIP.value
I_RHIP = LM.RIGHT_HIP.value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ball-replacer")


# ----------------------------- Helpers --------------------------------------

def safe_landmark(landmarks, idx: int) -> Optional[tuple[float, float, float]]:
    if landmarks is None:
        return None
    lm = landmarks[idx]
    return (float(lm.x), float(lm.y), float(lm.visibility))


def avg_xy(a, b) -> Optional[tuple[float, float]]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def create_ball_tracker(weights_path: str, nano_backbone: str, nano_neckhead: str):
    """Same configuration as 30_generate_debug.py / 10_build_dribble_dataset.py."""
    params = cv2.BallTrackerParams()
    params.yoloModelCalibration = weights_path
    params.yoloModelDetection = weights_path
    params.yoloImgszCalibration = 640
    params.yoloImgszDetection = 640
    params.yoloConfidence = 0.25
    params.nanoBackbone = nano_backbone
    params.nanoNeckhead = nano_neckhead
    params.searchCrops = 1
    params.numTemplates = 1
    params.calibrationFrames = 100
    params.yoloPeriodic = 3
    params.maxBboxJump = 100.0
    params.driftThreshold = 0.50
    return cv2.BallTracker.create(params)


# ----------------------------- Per-video processing -------------------------

def process_video_replace_ball(video_path: Path,
                               num_rows: int,
                               weights: str, nano_backbone: str, nano_neckhead: str,
                               pose) -> dict[int, dict]:
    """Run BallTracker + MediaPipe pose over `video_path` and return a dict
    mapping frame_id -> {Rel_Ball_X, Rel_Ball_Y, Ball_Detected}.

    `num_rows` is the number of frames the original CSV has for this video,
    used only for sanity-check logging.
    """
    tracker = create_ball_tracker(weights, nano_backbone, nano_neckhead)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Could not open %s", video_path)
        return {}

    updates: dict[int, dict] = {}
    frame_id = 0
    found_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1. Ball detection via BallTracker
        result = tracker.processFrame(frame)
        if result.found:
            cx_norm, cy_norm, _, _ = result.normBbox
            ball_norm = (float(cx_norm), float(cy_norm))
            ball_detected = 1.0
            found_count += 1
        else:
            ball_norm = None
            ball_detected = 0.0

        # 2. Hip center via MediaPipe (needed to make the ball coord relative)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        landmarks = pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None

        rel_ball_x = np.nan
        rel_ball_y = np.nan
        if ball_norm is not None and landmarks is not None:
            lhip = safe_landmark(landmarks, I_LHIP)
            rhip = safe_landmark(landmarks, I_RHIP)
            hip_center = avg_xy(lhip, rhip)
            if hip_center is not None:
                rel_ball_x = ball_norm[0] - hip_center[0]
                rel_ball_y = ball_norm[1] - hip_center[1]

        updates[frame_id] = {
            "Rel_Ball_X": rel_ball_x,
            "Rel_Ball_Y": rel_ball_y,
            "Ball_Detected": ball_detected,
        }
        frame_id += 1

    cap.release()
    log.info(
        "Processed %s: %d frames (CSV has %d rows), ball found in %d (%.1f%%)",
        video_path.name, frame_id, num_rows, found_count,
        100.0 * found_count / max(frame_id, 1),
    )
    return updates


# ----------------------------- CSV-level driver ------------------------------

def replace_ball_in_csv(csv_path: Path, out_path: Path, videos_dir: Path,
                        weights: str, nano_backbone: str, nano_neckhead: str) -> None:
    log.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)

    # Sort by (Video_ID, Frame_ID) so frame_id matches the cap.read() order.
    df = df.sort_values(["Video_ID", "Frame_ID"]).reset_index(drop=True)

    # Make sure Ball_Detected column exists. If new CSVs were generated by the
    # updated 10_build_dribble_dataset.py it will already be there.
    if "Ball_Detected" not in df.columns:
        df["Ball_Detected"] = 0.0

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        unique_videos = df["Video_ID"].unique()
        log.info("Found %d unique videos in %s", len(unique_videos), csv_path.name)

        for video_id in unique_videos:
            video_path = videos_dir / video_id
            if not video_path.exists():
                log.warning("Video file not found: %s -- skipping", video_path)
                continue

            group_count = int((df["Video_ID"] == video_id).sum())
            updates = process_video_replace_ball(
                video_path, group_count,
                weights, nano_backbone, nano_neckhead, pose,
            )
            if not updates:
                continue

            # Build an index of (Video_ID, Frame_ID) for fast updates.
            video_mask = df["Video_ID"] == video_id
            video_df = df.loc[video_mask]

            # Map frame_id -> dataframe row index for this video.
            frame_to_idx = dict(zip(video_df["Frame_ID"].astype(int).tolist(), video_df.index.tolist()))

            for frame_id, vals in updates.items():
                row_idx = frame_to_idx.get(frame_id)
                if row_idx is None:
                    # Frame exists in the video but not in the CSV (shouldn't
                    # normally happen — every frame should have a row).
                    continue
                df.at[row_idx, "Rel_Ball_X"] = vals["Rel_Ball_X"]
                df.at[row_idx, "Rel_Ball_Y"] = vals["Rel_Ball_Y"]
                df.at[row_idx, "Ball_Detected"] = vals["Ball_Detected"]
    finally:
        pose.close()

    # Recompute ball-to-wrist distances using the new ball + existing wrists.
    log.info("Recomputing Dist_Ball_L_Wrist / Dist_Ball_R_Wrist")
    df["Dist_Ball_L_Wrist"] = np.sqrt(
        (df["Rel_Ball_X"] - df["Rel_LeftWrist_X"]) ** 2
        + (df["Rel_Ball_Y"] - df["Rel_LeftWrist_Y"]) ** 2
    )
    df["Dist_Ball_R_Wrist"] = np.sqrt(
        (df["Rel_Ball_X"] - df["Rel_RightWrist_X"]) ** 2
        + (df["Rel_Ball_Y"] - df["Rel_RightWrist_Y"]) ** 2
    )

    # Recompute Delta_Ball_X/Y from a forward-filled ball position so missed
    # detections produce zero motion (matches the contract in 10_build_*).
    log.info("Recomputing Delta_Ball_X / Delta_Ball_Y (per-video, forward-filled)")
    ball_x_ff = df.groupby("Video_ID")["Rel_Ball_X"].ffill()
    ball_y_ff = df.groupby("Video_ID")["Rel_Ball_Y"].ffill()
    df["Delta_Ball_Y"] = ball_y_ff.groupby(df["Video_ID"]).diff().fillna(0.0)
    df["Delta_Ball_X"] = ball_x_ff.groupby(df["Video_ID"]).diff().fillna(0.0)

    df["Ball_Detected"] = df["Ball_Detected"].fillna(0.0)

    # Reorder columns to the canonical CSV layout. Columns missing from the
    # input CSV are added as NaN; any extra columns are dropped.
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
    extra = [c for c in df.columns if c not in CSV_COLUMNS]
    if extra:
        log.info("Dropping %d extra column(s) not in CSV_COLUMNS: %s", len(extra), extra)
    df = df[CSV_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %d rows -> %s", len(df), out_path)


# ----------------------------- Entry point ----------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv",   default="../data/bskt/datasets/training_dataset_smooth.csv")
    ap.add_argument("--validation-csv", default="../data/bskt/datasets/validation_dataset_smooth.csv")
    ap.add_argument("--training-out",   default=None,
                    help="Output path for the updated training CSV "
                         "(default: <input>_tracker.csv).")
    ap.add_argument("--validation-out", default=None,
                    help="Output path for the updated validation CSV "
                         "(default: <input>_tracker.csv).")
    ap.add_argument("--videos-dir", default="../raw_dribbling_videos/processed",
                    help="Directory containing the source .mp4 files. The "
                         "Video_ID column in the CSV is used as the file name.")
    ap.add_argument("--weights", default="../models/ball_detection_v26n_640_07_04_raw.onnx",
                    help="YOLO weights for BallTracker (must be a format "
                         "cv::dnn::readNet supports, e.g. _raw.onnx).")
    ap.add_argument("--nano-backbone", default="../models/nanotrack_backbone_sim_v2.onnx")
    ap.add_argument("--nano-neckhead", default="../models/nanotrack_head_sim_v2.onnx")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        raise SystemExit(f"Videos directory not found: {videos_dir}")

    def out_for(input_path: str, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        p = Path(input_path)
        return p.with_name(p.stem + "_tracker" + p.suffix)

    training_csv = Path(args.training_csv)
    validation_csv = Path(args.validation_csv)
    training_out = out_for(args.training_csv, args.training_out)
    validation_out = out_for(args.validation_csv, args.validation_out)

    if training_csv.exists():
        replace_ball_in_csv(training_csv, training_out, videos_dir,
                            args.weights, args.nano_backbone, args.nano_neckhead)
    else:
        log.warning("Training CSV not found: %s -- skipping", training_csv)

    if validation_csv.exists():
        replace_ball_in_csv(validation_csv, validation_out, videos_dir,
                            args.weights, args.nano_backbone, args.nano_neckhead)
    else:
        log.warning("Validation CSV not found: %s -- skipping", validation_csv)


if __name__ == "__main__":
    main()
