"""
Basketball dribble dataset builder.

Processes .mp4 files in ../raw_dribbling_videos, detects the ball with a YOLO
model, extracts MediaPipe pose landmarks, normalizes every coordinate to the
hip center, then runs the trained GRU world-model to fill the
Dribble / Crossover / Hand_Touch labels and writes training_data.csv.

Run:
    python build_dribble_dataset.py                 # headless
    python build_dribble_dataset.py --debug         # with cv2.imshow overlay
    python build_dribble_dataset.py --weights best.pt --videos ../raw_dribbling_videos
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO


# ----------------------------- Configuration --------------------------------

# Must match window_size used when training the GRU (see 5_build_gru_model.py).
INFERENCE_WINDOW = 15

# Pose-landmark index helpers (MediaPipe Pose, 33-point model).
LM = mp.solutions.pose.PoseLandmark
I_LSHOULDER = LM.LEFT_SHOULDER.value
I_RSHOULDER = LM.RIGHT_SHOULDER.value
I_LELBOW    = LM.LEFT_ELBOW.value
I_RELBOW    = LM.RIGHT_ELBOW.value
I_LHIP      = LM.LEFT_HIP.value
I_RHIP      = LM.RIGHT_HIP.value
I_LKNEE     = LM.LEFT_KNEE.value
I_RKNEE     = LM.RIGHT_KNEE.value
I_LANKLE    = LM.LEFT_ANKLE.value
I_RANKLE    = LM.RIGHT_ANKLE.value
I_LWRIST    = LM.LEFT_WRIST.value
I_RWRIST    = LM.RIGHT_WRIST.value

CSV_COLUMNS = [
    "Video_ID", "Frame_ID", "Norm_Torso_Height",
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y", "LeftElbow_Vis",
    "Rel_RightElbow_X", "Rel_RightElbow_Y", "RightElbow_Vis",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "Rel_LeftWrist_Z", "LeftWrist_Vis",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "Rel_RightWrist_Z", "RightWrist_Vis",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y", "LeftAnkle_Vis",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y", "RightAnkle_Vis",
    "Dist_Ball_L_Wrist", "Dist_Ball_R_Wrist",
    "Delta_Ball_Y", "Delta_Ball_X",
    "Left_Wrist_Behind", "Right_Wrist_Behind", "Hands_Behind_Back_Count",
    "Dribble", "Crossover", "Hand_Touch",
    "Ball_Detected",
]

# Order MUST match HoopsDataset.feature_cols in 60_build_gru_model.py.
# Ball_Detected is appended at the end (1.0 if YOLO/tracker found the ball this
# frame, 0.0 otherwise). This lets the GRU distinguish a real (0,0) ball
# position from a missed detection that was filled with 0.
FEATURE_COLS = [
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y", "LeftElbow_Vis",
    "Rel_RightElbow_X", "Rel_RightElbow_Y", "RightElbow_Vis",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "Rel_LeftWrist_Z", "LeftWrist_Vis",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "Rel_RightWrist_Z", "RightWrist_Vis",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y", "LeftAnkle_Vis",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y", "RightAnkle_Vis",
    "Norm_Torso_Height",
    "Dist_Ball_L_Wrist", "Dist_Ball_R_Wrist",
    "Delta_Ball_Y", "Delta_Ball_X",
    "Left_Wrist_Behind", "Right_Wrist_Behind", "Hands_Behind_Back_Count",
    "Ball_Detected",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("dribble-dataset")


# ----------------------------- GRU world-model -----------------------------

class HoopsWorldModel(nn.Module):
    """Architecture must match 5_build_gru_model.py exactly so the saved
    state_dict loads cleanly."""

    def __init__(self, input_size: int = 31, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_summary = gru_out[:, -1, :]
        return self.classifier_head(final_summary), self.predictor_head(final_summary)


def predict_action_labels(df: pd.DataFrame, model: HoopsWorldModel,
                          device: torch.device,
                          dribble_thr: float, crossover_thr: float,
                          touch_thr: float, batch_size: int = 256) -> pd.DataFrame:
    """Fill Dribble / Crossover / Hand_Touch from the GRU's classifier head.

    Mirrors the training contract from 5_build_gru_model.py: for every
    INFERENCE_WINDOW-frame slice the output is assigned to the LAST frame of
    that window (`y_actions = actions[i + window_size - 1]`). Frames with fewer
    than INFERENCE_WINDOW preceding frames inside their video stay 0.
    """
    df["Dribble"] = 0
    df["Crossover"] = 0
    df["Hand_Touch"] = 0

    feats = df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)

    model.eval()
    for video_id, group in df.groupby("Video_ID", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < INFERENCE_WINDOW:
            log.warning("Skipping %s: only %d frames (< %d).",
                        video_id, len(idx), INFERENCE_WINDOW)
            continue

        seq = feats[idx]  # (L, F) in original frame order
        # sliding_window_view yields (L-W+1, F, W); transpose to (L-W+1, W, F).
        windows = np.lib.stride_tricks.sliding_window_view(
            seq, window_shape=INFERENCE_WINDOW, axis=0
        ).transpose(0, 2, 1)
        windows = np.ascontiguousarray(windows)

        probs_chunks = []
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                chunk = torch.from_numpy(windows[start:start + batch_size]).to(device)
                p, _ = model(chunk)
                probs_chunks.append(p.cpu().numpy())
        probs = np.concatenate(probs_chunks, axis=0)

        target_idx = idx[INFERENCE_WINDOW - 1:]
        df.loc[target_idx, "Dribble"]    = (probs[:, 0] >= dribble_thr).astype(int)
        df.loc[target_idx, "Crossover"]  = (probs[:, 1] >= crossover_thr).astype(int)
        df.loc[target_idx, "Hand_Touch"] = (probs[:, 2] >= touch_thr).astype(int)

    return df


# ----------------------------- Helpers --------------------------------------

def inject_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ball-to-wrist distances and per-video ball velocity columns.

    Delta_Ball_{X,Y} are computed from a *forward-filled* ball position so a
    missed detection produces zero motion (consistent with "we assume the ball
    didn't move") rather than a NaN that gets silently mapped to 0. The GRU
    sees Ball_Detected separately, so it can decide how much to trust the
    forward-filled value.
    """
    df["Dist_Ball_L_Wrist"] = np.sqrt(
        (df["Rel_Ball_X"] - df["Rel_LeftWrist_X"]) ** 2
        + (df["Rel_Ball_Y"] - df["Rel_LeftWrist_Y"]) ** 2
    )
    df["Dist_Ball_R_Wrist"] = np.sqrt(
        (df["Rel_Ball_X"] - df["Rel_RightWrist_X"]) ** 2
        + (df["Rel_Ball_Y"] - df["Rel_RightWrist_Y"]) ** 2
    )

    ball_x_ff = df.groupby("Video_ID")["Rel_Ball_X"].ffill()
    ball_y_ff = df.groupby("Video_ID")["Rel_Ball_Y"].ffill()
    df["Delta_Ball_Y"] = ball_y_ff.groupby(df["Video_ID"]).diff().fillna(0.0)
    df["Delta_Ball_X"] = ball_x_ff.groupby(df["Video_ID"]).diff().fillna(0.0)

    # Discrete "wrist is behind the hip plane" indicators. MediaPipe's z is
    # negative in front of the camera and positive behind, anchored at the
    # hip mid-point, so a positive value past a small threshold means the
    # wrist has actually crossed behind the player's torso. The 0.05 threshold
    # ignores noise around z = 0 from imperfect pose tracking.
    df["Left_Wrist_Behind"]       = (df["Rel_LeftWrist_Z"]  > 0.05).astype(float)
    df["Right_Wrist_Behind"]      = (df["Rel_RightWrist_Z"] > 0.05).astype(float)
    df["Hands_Behind_Back_Count"] = df["Left_Wrist_Behind"] + df["Right_Wrist_Behind"]

    # Make sure missing detections have explicit zero rather than NaN.
    df["Ball_Detected"] = df["Ball_Detected"].fillna(0.0)
    return df


def safe_landmark(landmarks, idx: int) -> Optional[tuple[float, float, float, float]]:
    """Return (x, y, z, visibility) or None if the landmark is missing.

    MediaPipe's z is already expressed relative to the hip mid-point (negative
    in front of the camera, positive behind), so it does NOT need to be
    re-anchored against the hip center the way x/y are.
    """
    if landmarks is None:
        return None
    lm = landmarks[idx]
    # We removed the visibility filter here so the raw confidence score
    # gets saved to the CSV for the AI to learn from!
    return (float(lm.x), float(lm.y), float(lm.z), float(lm.visibility))


def avg_xy(a, b) -> Optional[tuple[float, float]]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def detect_ball(model: YOLO, frame_bgr: np.ndarray,
                frame_w: int, frame_h: int) -> Optional[tuple[float, float]]:
    """Run YOLO and return the (x, y) CENTER of the highest-conf class-0 box,
    normalized to [0,1]. Returns None if no ball was found."""
    results = model(frame_bgr, classes=[0], verbose=False)
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    # Pick highest-confidence detection.
    confs = r.boxes.conf.cpu().numpy()
    xyxy  = r.boxes.xyxy.cpu().numpy()
    best  = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[best]
    cx = (x1 + x2) * 0.5 / frame_w
    cy = (y1 + y2) * 0.5 / frame_h
    return (float(cx), float(cy))


def create_ball_tracker(weights_path: str, nano_backbone: str, nano_neckhead: str):
    """Create a cv2.BallTracker configured to use `weights_path` for both
    calibration and detection, plus the TrackerNano backbone/neckhead. The
    weights MUST be in a format cv::dnn::readNet supports (e.g. _raw.onnx)."""
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


def detect_ball_with_tracker(tracker, frame_bgr: np.ndarray) -> Optional[tuple[float, float]]:
    """Process a frame through cv2.BallTracker and return the (x, y) CENTER
    of the ball normalized to [0,1], or None if not found.

    BallTracker.processFrame already returns normBbox (cx, cy, w, h) in 0..1
    relative to the input frame.
    """
    result = tracker.processFrame(frame_bgr)
    if not result.found:
        return None
    cx, cy, _, _ = result.normBbox
    return (float(cx), float(cy))


# ----------------------------- Main per-video -------------------------------

def process_video(video_path: Path,
                  yolo: Optional[YOLO],
                  tracker,
                  pose: "mp.solutions.pose.Pose",
                  debug: bool) -> list[dict]:
    """Process a single video. Exactly one of `yolo` / `tracker` must be set.
    `tracker` is expected to be a fresh cv2.BallTracker instance (stateful)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Could not open %s", video_path)
        return []

    rows: list[dict] = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        if tracker is not None:
            ball_norm = detect_ball_with_tracker(tracker, frame)
        else:
            ball_norm = detect_ball(yolo, frame, w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        landmarks = pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None

        row = {c: np.nan for c in CSV_COLUMNS}
        row["Video_ID"] = video_path.name
        row["Frame_ID"] = frame_id
        # Tells the GRU whether Rel_Ball_X/Y are real or filled-in zeros.
        row["Ball_Detected"] = 1.0 if ball_norm is not None else 0.0
        # Action labels are filled later by the GRU; default to 0 so frames
        # without enough history (first INFERENCE_WINDOW-1 of each video) are
        # well-defined.
        row["Dribble"] = 0
        row["Crossover"] = 0
        row["Hand_Touch"] = 0

        hip_center = None

        if landmarks is not None:
            lhip = safe_landmark(landmarks, I_LHIP)
            rhip = safe_landmark(landmarks, I_RHIP)
            hip_center = avg_xy(lhip, rhip)

            lsh = safe_landmark(landmarks, I_LSHOULDER)
            rsh = safe_landmark(landmarks, I_RSHOULDER)
            shoulder_center = avg_xy(lsh, rsh)

            lelb = safe_landmark(landmarks, I_LELBOW)
            relb = safe_landmark(landmarks, I_RELBOW)
            lwr  = safe_landmark(landmarks, I_LWRIST)
            rwr  = safe_landmark(landmarks, I_RWRIST)
            lank = safe_landmark(landmarks, I_LANKLE)
            rank = safe_landmark(landmarks, I_RANKLE)

            if hip_center is not None:
                hcx, hcy = hip_center

                if shoulder_center is not None:
                    row["Norm_Torso_Height"] = abs(shoulder_center[1] - hcy)

                if ball_norm is not None:
                    row["Rel_Ball_X"] = ball_norm[0] - hcx
                    row["Rel_Ball_Y"] = ball_norm[1] - hcy

                if lelb is not None:
                    row["Rel_LeftElbow_X"]  = lelb[0] - hcx
                    row["Rel_LeftElbow_Y"]  = lelb[1] - hcy
                    row["LeftElbow_Vis"]    = lelb[3]
                if relb is not None:
                    row["Rel_RightElbow_X"] = relb[0] - hcx
                    row["Rel_RightElbow_Y"] = relb[1] - hcy
                    row["RightElbow_Vis"]   = relb[3]

                if lwr is not None:
                    row["Rel_LeftWrist_X"]  = lwr[0] - hcx
                    row["Rel_LeftWrist_Y"]  = lwr[1] - hcy
                    # MediaPipe's z is already hip-relative; do not subtract hcz.
                    row["Rel_LeftWrist_Z"]  = lwr[2]
                    row["LeftWrist_Vis"]    = lwr[3]
                if rwr is not None:
                    row["Rel_RightWrist_X"] = rwr[0] - hcx
                    row["Rel_RightWrist_Y"] = rwr[1] - hcy
                    row["Rel_RightWrist_Z"] = rwr[2]
                    row["RightWrist_Vis"]   = rwr[3]

                if lank is not None:
                    row["Rel_LeftAnkle_X"]  = lank[0] - hcx
                    row["Rel_LeftAnkle_Y"]  = lank[1] - hcy
                    row["LeftAnkle_Vis"]    = lank[3]
                if rank is not None:
                    row["Rel_RightAnkle_X"] = rank[0] - hcx
                    row["Rel_RightAnkle_Y"] = rank[1] - hcy
                    row["RightAnkle_Vis"]   = rank[3]
        else:
            log.debug("No pose in %s frame %d", video_path.name, frame_id)

        rows.append(row)

        if debug:
            dbg = frame.copy()
            if hip_center is not None:
                hx, hy = int(hip_center[0] * w), int(hip_center[1] * h)
                cv2.circle(dbg, (hx, hy), 8, (0, 255, 255), -1)
                cv2.putText(dbg, "HIP", (hx + 10, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if ball_norm is not None:
                    bx, by = int(ball_norm[0] * w), int(ball_norm[1] * h)
                    cv2.circle(dbg, (bx, by), 10, (0, 0, 255), 2)
                    cv2.line(dbg, (hx, hy), (bx, by), (0, 255, 0), 2)
            cv2.imshow("dribble-debug", dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Debug quit requested.")
                break

        frame_id += 1

    cap.release()
    log.info("Processed %s: %d frames, %d rows.", video_path.name, frame_id, len(rows))
    return rows


# ----------------------------- Entry point ----------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos",  default="../raw_dribbling_videos",
                    help="Directory of .mp4 inputs.")
    ap.add_argument("--weights", default="../models/ball_detection_v26n_640_07_04.pt",
                    help="YOLO weights (class 0 = ball).")
    ap.add_argument("--gru-weights", default="../models/hoops_world_model_best.pth",
                    help="Trained GRU world-model state_dict (.pth).")
    ap.add_argument("--dribble-thr",   type=float, default=0.5)
    ap.add_argument("--crossover-thr", type=float, default=0.5)
    ap.add_argument("--touch-thr",     type=float, default=0.5)
    ap.add_argument("--out",     default="../data/bskt/training_data.csv")
    ap.add_argument("--debug",   action="store_true",
                    help="Show cv2.imshow overlay with hip anchor + ball line.")
    ap.add_argument("--tracker", action="store_true",
                    help="Use the custom cv2.BallTracker (YOLO + TrackerNano) "
                         "instead of stand-alone YOLO. Requires --nano-backbone "
                         "and --nano-neckhead, and --weights must be a format "
                         "cv::dnn::readNet supports (e.g. _raw.onnx).")
    ap.add_argument("--nano-backbone", default=None,
                    help="Path to TrackerNano backbone ONNX (required with --tracker).")
    ap.add_argument("--nano-neckhead", default=None,
                    help="Path to TrackerNano neckhead ONNX (required with --tracker).")
    ap.add_argument("--no-gru", action="store_true",
                    help="Skip the final GRU labeling step (Dribble / Crossover / "
                         "Hand_Touch stay 0). Useful when generating CSVs to "
                         "train a new GRU.")
    args = ap.parse_args()

    if args.tracker and (not args.nano_backbone or not args.nano_neckhead):
        ap.error("--tracker requires both --nano-backbone and --nano-neckhead")

    videos_dir = Path(args.videos)
    if not videos_dir.is_dir():
        raise SystemExit(f"Videos directory not found: {videos_dir}")

    mp4s = sorted(videos_dir.glob("*.mp4"))
    if not mp4s:
        raise SystemExit(f"No .mp4 files in {videos_dir}")

    yolo = None
    if not args.tracker:
        log.info("Loading YOLO weights from %s", args.weights)
        yolo = YOLO(args.weights)
    else:
        log.info("Using BallTracker (weights=%s, backbone=%s, neckhead=%s)",
                 args.weights, args.nano_backbone, args.nano_neckhead)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_rows: list[dict] = []
    try:
        for vp in mp4s:
            # Create a fresh tracker per video so calibration / color model /
            # recovery state restart for each clip.
            tracker = None
            if args.tracker:
                tracker = create_ball_tracker(
                    args.weights, args.nano_backbone, args.nano_neckhead
                )
            all_rows.extend(process_video(vp, yolo, tracker, pose, debug=args.debug))
    finally:
        pose.close()
        if args.debug:
            cv2.destroyAllWindows()

    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    df = inject_derived_features(df)

    if args.no_gru:
        log.info("--no-gru set; skipping GRU labeling step "
                 "(Dribble/Crossover/Hand_Touch stay 0).")
    else:
        log.info("Loading GRU weights from %s", args.gru_weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gru = HoopsWorldModel(input_size=len(FEATURE_COLS))
        gru.load_state_dict(torch.load(args.gru_weights, map_location=device, weights_only=True))
        gru.to(device).eval()

        log.info("Predicting Dribble/Crossover/Hand_Touch with GRU on %s...", device)
        df = predict_action_labels(
            df, gru, device,
            dribble_thr=args.dribble_thr,
            crossover_thr=args.crossover_thr,
            touch_thr=args.touch_thr,
        )

    df.to_csv(args.out, index=False)
    log.info("Wrote %d rows -> %s", len(df), args.out)


if __name__ == "__main__":
    main()