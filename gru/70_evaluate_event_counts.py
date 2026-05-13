"""Event-count evaluator: runs the trained classifier across each video in a
labeled CSV (training or validation), applies the same hysteresis + crossover
lookahead as 90_test_inference.py, and reports predicted vs. ground-truth event
counts per video plus aggregate accuracy.

This is the metric that actually matches the production goal (counting dribbles
and crossovers). Frame-level F1 is a proxy; event-count accuracy is the truth.

Run:  python 70_evaluate_event_counts.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque

# ==========================================
# 1. MODEL CLASS (Must match training architecture exactly)
# ==========================================
class HoopsWorldModel(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, num_layers=2):
        super(HoopsWorldModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
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


# ==========================================
# 2. FEATURES (Must match training exactly — same 31 columns, same order)
# ==========================================
FEATURE_COLS = [
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y", "LeftElbow_Vis",
    "Rel_RightElbow_X", "Rel_RightElbow_Y", "RightElbow_Vis",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "Rel_LeftWrist_Z", "LeftWrist_Vis",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "Rel_RightWrist_Z", "RightWrist_Vis",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y", "LeftAnkle_Vis",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y", "RightAnkle_Vis",
    "Norm_Torso_Height",
    "Dist_Ball_L_Wrist",
    "Dist_Ball_R_Wrist",
    "Delta_Ball_Y",
    "Delta_Ball_X",
    "Left_Wrist_Behind",
    "Right_Wrist_Behind",
    "Hands_Behind_Back_Count",
    "Ball_Detected",
]


# ==========================================
# 3. EVENT COUNTING — GROUND TRUTH AND PREDICTIONS
# ==========================================
def count_events_in_labels(soft_labels, threshold=0.5):
    """Count distinct events from a soft-label series. An event is one
    contiguous run of frames where label >= threshold. Counts the number
    of rising edges (0 -> 1 transitions in the binarized signal), plus 1 if
    the video starts already in a positive run."""
    binarized = (np.asarray(soft_labels) >= threshold).astype(np.int8)
    if len(binarized) == 0:
        return 0
    edges = np.diff(binarized)
    rising = int((edges == 1).sum())
    starts_high = int(binarized[0] == 1)
    return rising + starts_high


def predict_video(features, model, window_size=15):
    """Slide a window over the feature stream and run the model on each full
    window. Returns a list of length len(features); entries before the window
    fills are None (not enough context to predict)."""
    probs = [None] * len(features)
    window = deque(maxlen=window_size)
    for idx, feat in enumerate(features):
        window.append(feat)
        if len(window) == window_size:
            x = torch.FloatTensor(np.array(window)).unsqueeze(0)
            with torch.no_grad():
                pred_actions, _ = model(x)
            probs[idx] = pred_actions[0].numpy()
    return probs


def count_predicted_events(probs,
                           dribble_trigger=0.50, dribble_reset=0.30,
                           crossover_threshold=0.30, crossover_lookahead=5):
    """Apply the same hysteresis + crossover-lookahead state machine as
    90_test_inference.py. Returns (total_dribbles, total_crossovers).
    Keep these defaults in sync with the inference script so this evaluator
    measures what gets shipped."""
    total_dribbles = 0
    total_crossovers = 0
    is_dribbling = False
    pending = None  # {'frames_left', 'max_p'} for crossover lookahead

    for p in probs:
        if p is None:
            continue
        p_dribble, p_crossover, _ = float(p[0]), float(p[1]), float(p[2])

        # 1. Dribble trigger (open new crossover window, finalize any open one)
        if p_dribble > dribble_trigger and not is_dribbling:
            total_dribbles += 1
            is_dribbling = True
            if pending is not None and pending['max_p'] > crossover_threshold:
                total_crossovers += 1
            pending = {'frames_left': crossover_lookahead, 'max_p': p_crossover}

        # 2. Dribble release
        elif p_dribble < dribble_reset and is_dribbling:
            is_dribbling = False

        # 3. Update open crossover window
        if pending is not None:
            pending['max_p'] = max(pending['max_p'], p_crossover)
            pending['frames_left'] -= 1
            if pending['frames_left'] <= 0:
                if pending['max_p'] > crossover_threshold:
                    total_crossovers += 1
                pending = None

    # Finalize any leftover pending at video end
    if pending is not None and pending['max_p'] > crossover_threshold:
        total_crossovers += 1

    return total_dribbles, total_crossovers


def find_peaks_with_min_gap(values, threshold, min_gap):
    """Return indices of local maxima in `values` that are >= threshold, with
    each peak separated from the previous by at least min_gap frames. Within
    each contiguous above-threshold run, only the single highest sample is
    selected. Successive peaks closer than min_gap collapse to just the first
    — that's what suppresses oscillation over-counts."""
    peaks = []
    n = len(values)
    i = 0
    last_peak = -min_gap
    while i < n:
        if values[i] < threshold:
            i += 1
            continue
        max_idx = i
        max_val = values[i]
        j = i + 1
        while j < n and values[j] >= threshold:
            if values[j] > max_val:
                max_val = values[j]
                max_idx = j
            j += 1
        if max_idx - last_peak >= min_gap:
            peaks.append(max_idx)
            last_peak = max_idx
        i = j
    return peaks


def count_predicted_events_peaks(probs,
                                 dribble_threshold=0.50, dribble_min_gap=5,
                                 crossover_threshold=0.30, crossover_window=5):
    """Peak-detection alternative to count_predicted_events.

    For each local maximum of p_dribble >= dribble_threshold (separated by
    >= dribble_min_gap frames), count one dribble. For each dribble peak,
    count one crossover if max(p_crossover) within +/- crossover_window frames
    around the peak exceeds crossover_threshold.

    Avoids the two failure modes the hysteresis state machine is sensitive to:
    - Over-count from p_dribble oscillating around the trigger threshold
      (one real dribble counted as multiple events).
    - Under-count when p_dribble doesn't drop below the reset threshold
      between fast successive dribbles (lock stays engaged, only first counts).
    """
    first = next((i for i, p in enumerate(probs) if p is not None), None)
    if first is None:
        return 0, 0
    valid = probs[first:]
    p_dribble = np.array([p[0] for p in valid], dtype=np.float32)
    p_crossover = np.array([p[1] for p in valid], dtype=np.float32)

    dribble_peaks = find_peaks_with_min_gap(p_dribble, dribble_threshold, dribble_min_gap)
    total_dribbles = len(dribble_peaks)
    total_crossovers = 0
    n = len(p_crossover)
    for peak in dribble_peaks:
        lo = max(0, peak - crossover_window)
        hi = min(n, peak + crossover_window + 1)
        if p_crossover[lo:hi].max() > crossover_threshold:
            total_crossovers += 1
    return total_dribbles, total_crossovers


# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def evaluate(csv_path, model_path):
    print(f"Loading model from {model_path}...")
    model = HoopsWorldModel(input_size=31)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    print()
    header = f"{'Video':<45} | {'Dribble GT/Hys/Peak (Δh/Δp)':<32} | {'Crossover GT/Hys/Peak (Δh/Δp)'}"
    print(header)
    print("-" * len(header))

    rows = []
    for video_id, group in df.groupby('Video_ID'):
        group = group.sort_values('Frame_ID').reset_index(drop=True)
        features = group[FEATURE_COLS].values

        gt_dribbles = count_events_in_labels(group['Dribble'].values)
        gt_crossovers = count_events_in_labels(group['Crossover'].values)

        probs = predict_video(features, model)
        hys_dribbles, hys_crossovers = count_predicted_events(probs)
        peak_dribbles, peak_crossovers = count_predicted_events_peaks(probs)

        dh = hys_dribbles - gt_dribbles
        dp = peak_dribbles - gt_dribbles
        ch = hys_crossovers - gt_crossovers
        cp = peak_crossovers - gt_crossovers

        print(f"{str(video_id):<45} | "
              f"{gt_dribbles:>3}/{hys_dribbles:>3}/{peak_dribbles:<3} ({dh:+3d}/{dp:+3d})         | "
              f"{gt_crossovers:>3}/{hys_crossovers:>3}/{peak_crossovers:<3} ({ch:+3d}/{cp:+3d})")

        rows.append({
            'video': video_id,
            'gt_d': gt_dribbles, 'hys_d': hys_dribbles, 'peak_d': peak_dribbles,
            'gt_c': gt_crossovers, 'hys_c': hys_crossovers, 'peak_c': peak_crossovers,
        })

    print("-" * len(header))

    if not rows:
        print("No videos found in the CSV.")
        return

    # Aggregate stats
    total_gt_d = sum(r['gt_d'] for r in rows)
    total_hys_d = sum(r['hys_d'] for r in rows)
    total_peak_d = sum(r['peak_d'] for r in rows)
    total_gt_c = sum(r['gt_c'] for r in rows)
    total_hys_c = sum(r['hys_c'] for r in rows)
    total_peak_c = sum(r['peak_c'] for r in rows)

    hys_mae_d = float(np.mean([abs(r['hys_d'] - r['gt_d']) for r in rows]))
    peak_mae_d = float(np.mean([abs(r['peak_d'] - r['gt_d']) for r in rows]))
    hys_mae_c = float(np.mean([abs(r['hys_c'] - r['gt_c']) for r in rows]))
    peak_mae_c = float(np.mean([abs(r['peak_c'] - r['gt_c']) for r in rows]))

    # Aggregate accuracy on totals: 1 - |gt-pred| / max(gt, 1). Can mask
    # cancelling per-video errors, so MAE per video is the more honest number.
    def acc(gt, pred):
        return 1.0 - abs(gt - pred) / max(gt, 1)

    print(f"{'TOTAL':<45} | "
          f"{total_gt_d:>3}/{total_hys_d:>3}/{total_peak_d:<3}                      | "
          f"{total_gt_c:>3}/{total_hys_c:>3}/{total_peak_c:<3}")
    print()
    print(f"{'Method':<14} | {'Dribble acc':<13} | {'Crossover acc':<15} | {'Dribble MAE/vid':<16} | {'Crossover MAE/vid'}")
    print("-" * 90)
    print(f"{'Hysteresis':<14} | {100 * acc(total_gt_d, total_hys_d):>10.1f}%   | "
          f"{100 * acc(total_gt_c, total_hys_c):>12.1f}%   | "
          f"{hys_mae_d:>13.2f}    | {hys_mae_c:>13.2f}")
    print(f"{'Peak detect':<14} | {100 * acc(total_gt_d, total_peak_d):>10.1f}%   | "
          f"{100 * acc(total_gt_c, total_peak_c):>12.1f}%   | "
          f"{peak_mae_d:>13.2f}    | {peak_mae_c:>13.2f}")
    print()
    print("Notes:")
    print("  - Aggregate accuracy hides per-video errors that cancel out across videos.")
    print("    MAE/vid is the more honest, conservative metric to ship against.")
    print("  - Hysteresis: state-machine counter (matches 90_test_inference.py exactly).")
    print("  - Peak detect: local-maxima counter, immune to oscillation over-count and")
    print("    hysteresis-lock under-count. Tunable via dribble_min_gap.")


if __name__ == "__main__":
    # --- CHANGE THESE PATHS TO MATCH YOUR FILES ---
    CSV_PATH = "../data/bskt/current_datasets/validation_dataset_smooth_tracker_z.csv"
    MODEL_PATH = "../models/hoops_world_model_best_f1.pth"

    evaluate(CSV_PATH, MODEL_PATH)
