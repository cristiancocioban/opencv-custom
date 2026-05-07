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
    def __init__(self, input_size=26, hidden_size=64, num_layers=2):
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
# 2. FEATURES (Must match training exactly — same 26 columns, same order)
# ==========================================
FEATURE_COLS = [
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y", "LeftElbow_Vis",
    "Rel_RightElbow_X", "Rel_RightElbow_Y", "RightElbow_Vis",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "LeftWrist_Vis",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "RightWrist_Vis",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y", "LeftAnkle_Vis",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y", "RightAnkle_Vis",
    "Norm_Torso_Height",
    "Dist_Ball_L_Wrist",
    "Dist_Ball_R_Wrist",
    "Delta_Ball_Y",
    "Delta_Ball_X",
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


# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def evaluate(csv_path, model_path):
    print(f"Loading model from {model_path}...")
    model = HoopsWorldModel(input_size=26)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    print()
    header = f"{'Video':<55} | {'Dribble GT/Pred (Δ)':<24} | {'Crossover GT/Pred (Δ)'}"
    print(header)
    print("-" * len(header))

    rows = []
    for video_id, group in df.groupby('Video_ID'):
        group = group.sort_values('Frame_ID').reset_index(drop=True)
        features = group[FEATURE_COLS].values

        gt_dribbles = count_events_in_labels(group['Dribble'].values)
        gt_crossovers = count_events_in_labels(group['Crossover'].values)

        probs = predict_video(features, model)
        pred_dribbles, pred_crossovers = count_predicted_events(probs)

        d_diff = pred_dribbles - gt_dribbles
        c_diff = pred_crossovers - gt_crossovers

        print(f"{str(video_id):<55} | {gt_dribbles:>4}/{pred_dribbles:<4} ({d_diff:+4d})       | "
              f"{gt_crossovers:>4}/{pred_crossovers:<4} ({c_diff:+4d})")

        rows.append({
            'video': video_id,
            'gt_d': gt_dribbles, 'pred_d': pred_dribbles, 'd_err': d_diff,
            'gt_c': gt_crossovers, 'pred_c': pred_crossovers, 'c_err': c_diff,
        })

    print("-" * len(header))

    if not rows:
        print("No videos found in the CSV.")
        return

    # Aggregate stats
    total_gt_d = sum(r['gt_d'] for r in rows)
    total_pred_d = sum(r['pred_d'] for r in rows)
    total_gt_c = sum(r['gt_c'] for r in rows)
    total_pred_c = sum(r['pred_c'] for r in rows)

    mae_d = float(np.mean([abs(r['d_err']) for r in rows]))
    mae_c = float(np.mean([abs(r['c_err']) for r in rows]))

    # Symmetric mean absolute percentage-style accuracy on totals.
    # 1 - |gt - pred| / max(gt, 1). Bounded above by 1.0; can go negative if
    # the model wildly over-predicts (e.g. predicts 50 when ground truth is 5).
    def acc(gt, pred):
        return 1.0 - abs(gt - pred) / max(gt, 1)

    print(f"{'TOTAL':<55} | {total_gt_d:>4}/{total_pred_d:<4} (MAE/vid {mae_d:.1f})  | "
          f"{total_gt_c:>4}/{total_pred_c:<4} (MAE/vid {mae_c:.1f})")
    print()
    print(f"Aggregate dribble count accuracy:    {100 * acc(total_gt_d, total_pred_d):.1f}%")
    print(f"Aggregate crossover count accuracy:  {100 * acc(total_gt_c, total_pred_c):.1f}%")
    print()
    print("Note: aggregate accuracy hides per-video errors that cancel out.")
    print("      MAE/vid is the average per-video absolute miscount and is the more")
    print("      conservative number to ship against.")


if __name__ == "__main__":
    # --- CHANGE THESE PATHS TO MATCH YOUR FILES ---
    CSV_PATH = "../data/bskt/current_datasets/validation_dataset_smooth_tracker.csv"
    MODEL_PATH = "../models/hoops_world_model_best_f1.pth"

    evaluate(CSV_PATH, MODEL_PATH)
