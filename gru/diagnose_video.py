"""
One-off diagnostic for a single Video_ID in the validation CSV. Loads the
trained model, runs it frame-by-frame, and reports:

  - Feature-side health: ball detection rate, pose visibility, NaN rates.
  - Model output: probability distributions and per-event hit rate at the
    peak frame.

Use --compare to print the same diagnostics for a well-counted clip
side-by-side, so anomalies stand out.

Run:
    python diagnose_video.py --video crossover_02_05_1.mp4 \\
                             --compare crossover_06_05_05.mp4
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


INPUT_SIZE = 31
WINDOW_SIZE = 15

# Must match training (60_build_gru_model.py).
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


class HoopsWorldModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 3), nn.Sigmoid(),
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final = gru_out[:, -1, :]
        return self.classifier_head(final), self.predictor_head(final)


def run_model(df_video, model):
    features = df_video[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
    probs = [None] * len(features)
    window = deque(maxlen=WINDOW_SIZE)
    for idx, feat in enumerate(features):
        window.append(feat)
        if len(window) == WINDOW_SIZE:
            x = torch.FloatTensor(np.array(window)).unsqueeze(0)
            with torch.no_grad():
                pred_actions, _ = model(x)
            probs[idx] = pred_actions[0].numpy()
    return probs


def find_gt_events(soft_labels, threshold=0.5):
    """Return list of (start, peak, end) frame indices for each contiguous
    above-threshold run."""
    binarized = (np.asarray(soft_labels) >= threshold).astype(np.int8)
    events = []
    in_event = False
    start = None
    for i, v in enumerate(binarized):
        if v and not in_event:
            in_event = True
            start = i
        elif not v and in_event:
            in_event = False
            peak = start + int(np.argmax(soft_labels[start:i]))
            events.append((start, peak, i - 1))
    if in_event:
        peak = start + int(np.argmax(soft_labels[start:]))
        events.append((start, peak, len(binarized) - 1))
    return events


def feature_health(df_video, video_id):
    print(f"\n=== Feature health: {video_id} ===")
    n = len(df_video)
    print(f"  Frames: {n}")
    bd_rate = df_video["Ball_Detected"].fillna(0).mean()
    print(f"  Ball_Detected rate: {bd_rate:.1%}")
    for col in ["LeftElbow_Vis", "RightElbow_Vis",
                "LeftWrist_Vis", "RightWrist_Vis",
                "LeftAnkle_Vis", "RightAnkle_Vis"]:
        v = df_video[col].dropna()
        if len(v):
            print(f"  {col}: mean={v.mean():.2f} min={v.min():.2f} "
                  f"<0.5: {(v < 0.5).mean():.1%}")
    h = df_video["Norm_Torso_Height"].dropna()
    if len(h):
        print(f"  Norm_Torso_Height: mean={h.mean():.3f} std={h.std():.3f}")
    # Ball motion stats — a stuck/spiky tracker shows up here.
    dx = df_video["Delta_Ball_X"].fillna(0)
    dy = df_video["Delta_Ball_Y"].fillna(0)
    print(f"  |Delta_Ball_X|: mean={dx.abs().mean():.3f} max={dx.abs().max():.3f}")
    print(f"  |Delta_Ball_Y|: mean={dy.abs().mean():.3f} max={dy.abs().max():.3f}")
    nan_rates = df_video[FEATURE_COLS].isna().mean()
    high_nan = nan_rates[nan_rates > 0.05]
    if len(high_nan):
        print(f"  Features with >5% NaN: {high_nan.to_dict()}")


def event_diagnostics(df_video, probs, action_col, action_idx,
                      fire_threshold=0.50):
    labels = df_video[action_col].fillna(0.0).to_numpy()
    events = find_gt_events(labels)
    if not events:
        return
    fired = 0
    misses = []
    peak_probs = []
    for start, peak, end in events:
        p_at_peak = probs[peak]
        if p_at_peak is None:
            continue
        p = float(p_at_peak[action_idx])
        peak_probs.append(p)
        if p >= fire_threshold:
            fired += 1
        else:
            misses.append((peak, p))
    if peak_probs:
        peak_arr = np.array(peak_probs)
        print(f"  {action_col}: {len(events)} GT events, "
              f"{fired}/{len(events)} fired at peak (threshold {fire_threshold}). "
              f"Peak-frame prob: mean={peak_arr.mean():.3f} "
              f"median={np.median(peak_arr):.3f} min={peak_arr.min():.3f}")
    if misses:
        misses.sort(key=lambda t: t[1])
        sample = misses[:8]
        print(f"    Lowest-prob misses: " +
              ", ".join(f"f{f}={p:.2f}" for f, p in sample))


def probs_summary(probs, video_id):
    valid = np.array([p for p in probs if p is not None])
    if len(valid) == 0:
        print(f"  (no full-window predictions for {video_id})")
        return
    print(f"\n=== Probability distribution: {video_id} ===")
    for i, name in enumerate(["Dribble", "Crossover", "Hand_Touch"]):
        p = valid[:, i]
        print(f"  {name}: mean={p.mean():.3f} median={np.median(p):.3f} "
              f"max={p.max():.3f}  frac>0.30: {(p > 0.30).mean():.1%}  "
              f"frac>0.50: {(p > 0.50).mean():.1%}")


def diagnose(df, video_id, model):
    df_video = df[df["Video_ID"] == video_id].sort_values("Frame_ID").reset_index(drop=True)
    if len(df_video) == 0:
        print(f"{video_id}: not found in CSV.")
        return
    feature_health(df_video, video_id)
    probs = run_model(df_video, model)
    probs_summary(probs, video_id)
    print(f"\n=== Per-event hit rate at peak frame (threshold 0.50): {video_id} ===")
    for col, idx in [("Dribble", 0), ("Crossover", 1)]:
        event_diagnostics(df_video, probs, col, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="Video_ID to diagnose (e.g. crossover_02_05_1.mp4).")
    ap.add_argument("--compare", default=None,
                    help="Optional second Video_ID for side-by-side comparison.")
    ap.add_argument("--csv",
                    default="../data/bskt/current_datasets/validation_dataset_smooth_tracker_z.csv",
                    help="Labeled CSV containing the video(s).")
    ap.add_argument("--model", default="../models/hoops_world_model_best_f1.pth",
                    help="Trained GRU checkpoint.")
    args = ap.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows, {df['Video_ID'].nunique()} unique videos")

    print(f"Loading model: {args.model}")
    model = HoopsWorldModel(input_size=INPUT_SIZE)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    diagnose(df, args.video, model)
    if args.compare:
        diagnose(df, args.compare, model)


if __name__ == "__main__":
    main()
