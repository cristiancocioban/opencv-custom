# Step 5 — Evaluate Event Counts

`70_evaluate_event_counts.py`

Measures the metric the system actually ships against: how accurately the
trained model counts dribbles and crossovers per video. Frame-level F1
from training is a useful proxy, but only event-count MAE tells you what
a downstream consumer (e.g. a stats dashboard) will see.

## What this script does

1. Loads a trained model checkpoint (defaults to `hoops_world_model_best_f1.pth`).
2. Loads a labeled CSV (defaults to `validation_dataset_smooth_tracker.csv`).
3. For each `Video_ID` group:
   - Counts ground-truth events from the soft labels (rising edges in
     `label ≥ 0.5`).
   - Runs the model frame-by-frame on the 31 features (sliding window
     of 15) to get per-frame probabilities.
   - Counts predicted events with **two** different counting algorithms,
     side by side: hysteresis (matches `90_test_inference.py`) and peak
     detection.
4. Prints per-video deltas and an aggregate comparison table including
   MAE per video and aggregate accuracy for each method.

## Two counting algorithms

Both methods take the same per-frame `(p_dribble, p_crossover, p_touch)`
from the model and produce `(total_dribbles, total_crossovers)`. They
differ in how they convert the probability stream into discrete events.

### A. Hysteresis (`count_predicted_events`)

State machine. Mirrors `90_test_inference.py` exactly so this evaluator
measures what the deployed inference script actually counts.

```
trigger:  p_dribble > 0.50 and not is_dribbling
           → total_dribbles++; lock engaged
release:  p_dribble < 0.30 and is_dribbling
           → lock released
```

For each dribble trigger, opens a 5-frame **crossover lookahead** window;
if `max(p_crossover)` over those 5 frames exceeds 0.30, increments the
crossover count too. If a new dribble triggers while a window is still
open, the in-flight window is finalized first (no silent loss).

**Failure modes:**
- *Over-count from threshold oscillation*: if `p_dribble` repeatedly
  crosses up through 0.5 → drops below 0.3 → crosses up again within a
  single real dribble, each up-crossing fires a count.
- *Under-count from lock-stuck*: if rapid successive dribbles never let
  `p_dribble` drop below 0.3, only the first is counted.

### B. Peak detection (`count_predicted_events_peaks`)

Local-maxima counter. Immune to both failure modes above.

```
1. For each contiguous run where p_dribble >= 0.50, the highest sample
   in the run is a candidate peak.
2. Successive peaks closer than `dribble_min_gap` (default 5) frames
   collapse to just the first — that suppresses oscillation over-counts.
3. For each surviving peak, count one crossover if max(p_crossover)
   within ±crossover_window (default 5) frames around the peak exceeds
   crossover_threshold (default 0.30).
```

**Why this works on rapid events:** the gap rule is between *peaks*, not
between threshold crossings. As long as two real fast dribbles produce
two distinct local maxima of `p_dribble` separated by ≥ 5 frames, both
are counted, regardless of whether the probability dipped below 0.3
between them.

**Tradeoff:** peak detection collapses peaks closer than `min_gap`. If
you have legitimate dribbles at >30Hz cadence (4 frames apart at 30fps),
they'll merge. Empirically `min_gap=5` is the sweet spot at 30fps —
small enough to resolve fast crossover sequences, large enough to absorb
prediction noise within a single event.

### Counting the GROUND TRUTH

For every label column in the labeled CSV:

```python
def count_events_in_labels(soft_labels, threshold=0.5):
    binarized = soft_labels >= threshold
    rising_edges = number of 0→1 transitions
    starts_high  = 1 if binarized[0] else 0
    return rising_edges + starts_high
```

Each contiguous above-threshold run = one event. This is the **definition
of "ground-truth event count"** that all per-video and aggregate numbers
in this evaluator measure against.

## Output format

```
Video                                | Dribble GT/Hys/Peak (Δh/Δp)   | Crossover GT/Hys/Peak (Δh/Δp)
crossover_06_05_01.mp4               |  32/41/36 (+9/+4)             |  32/41/36 (+9/+4)
crossover_06_05_02.mp4               |  10/ 9/ 9 (-1/-1)             |  10/ 9/ 9 (-1/-1)
...
TOTAL                                | 202/202/197                   | 167/172/168

Method      | Dribble acc | Crossover acc | Dribble MAE/vid | Crossover MAE/vid
Hysteresis  |    100.0%   |        97.0%  |          3.33   |          2.83
Peak detect |     97.5%   |        99.4%  |          2.50   |          1.83
```

### Reading the table

- **GT / Hys / Peak**: ground-truth count, hysteresis count, peak count
  for each video.
- **(Δh / Δp)**: hysteresis delta, peak delta. Negative = under-count;
  positive = over-count.
- **TOTAL**: column-wise sums. Useful for spotting whether one method
  systematically over- or under-counts.
- **Aggregate accuracy** (`1 − |GT − Pred| / max(GT, 1)`): shipping-style
  one-number summary. **But this can hide cancelling errors** — a video
  with +9 over-count and another with -9 under-count sum to perfect
  accuracy that hides a real problem. Don't optimize against this in
  isolation.
- **MAE/vid**: mean of absolute per-video deltas. The most honest
  conservative number to ship against. A model with MAE=3.0 misses ~3
  events per video, period — no error cancellation possible.

## Tunable parameters

All in `if __name__ == "__main__":` or as function defaults:

```python
# In count_predicted_events (hysteresis):
dribble_trigger      = 0.50   # p_dribble must exceed this to fire
dribble_reset        = 0.30   # p_dribble must drop below this before
                              # another fire is allowed
crossover_threshold  = 0.30   # max p_crossover over the lookahead must exceed this
crossover_lookahead  = 5      # frames after dribble trigger to evaluate

# In count_predicted_events_peaks:
dribble_threshold    = 0.50   # entry threshold for above-threshold runs
dribble_min_gap      = 5      # minimum frames between successive peaks
crossover_threshold  = 0.30   # same semantics as hysteresis
crossover_window     = 5      # ±N frames around each peak to scan p_crossover
```

If the inference script's hysteresis values change, update both function
defaults here — the script's docstring explicitly notes that the evaluator
should match `90_test_inference.py` so the offline number reflects what
ships.

## Recommended use during development

1. After every retraining, run `python 70_evaluate_event_counts.py`. The
   peak-detect MAE is the headline metric. If it improved by ≥ 0.5,
   ship the new model; otherwise the change wasn't worth it.
2. Drill into per-video outliers. Big positive Δp (over-count) usually
   means model probabilities are noisy on that clip — consider whether
   it's an unusual style or could indicate a feature-extraction bug.
   Big negative Δp (under-count) on otherwise-fine clips usually means
   `dribble_min_gap` is too aggressive for that player's cadence.
3. Compare hysteresis Δh and peak Δp side-by-side. If they agree, the
   model itself is the bottleneck (better data or architecture needed).
   If they disagree, the counter is the bottleneck (tune the parameters
   above).

## Production sync

`90_test_inference.py` (live overlay on a single video) and this script
should produce **the same per-video count** for the same model + video.
Both implement peak detection with identical parameters. If they
diverge, check:
- Is `90_test_inference.py` running with the same model file?
- Is its `DRIBBLE_MIN_GAP` set the same way?
- Is the live YOLO/MediaPipe re-extracting features that match what's
  in the CSV? Tracker state can differ slightly between runs of the same
  video, since `cv2.BallTracker` is re-initialized fresh.
