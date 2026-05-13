# Optional — Class-Targeted Augmentation

`55_augment_class.py`

An **optional** pass that triples the representation of one action class in
the smoothed training CSV by appending mirrored and mirrored+jittered copies
of every video whose filename contains the class name. Default class is
`Crossover`, the rarest of the three. Runs after step 3 (smoothing) and
produces a new CSV that step 4 (training) consumes in place of the un-
augmented one.

This step is not part of the canonical pipeline — skip it for a baseline
run, enable it when one class is consistently lagging in val F1.

## Why augment only one class

The training set is class-imbalanced. From [60_model_training.md](60_model_training.md):

> Crossover typically lags the other two because it's the rarest class
> (~7% positives in training).

Two ways to attack imbalance: change the loss (class-weighted BCE, focal
loss), or change the data (oversample the rare class). We chose the data
side because:

- The loss is already class-aware in spirit — the multi-head structure
  computes per-class BCE.
- Oversampling preserves the calibration property of `BCELoss` on soft
  targets, which is the whole reason the smoother in step 3 exists.
- Crossover examples are physically symmetric: a R→L crossover and a L→R
  crossover are mirror images of the same move. That's free 2× data
  without a single new clip recorded.

## What gets duplicated

The script operates at the **video level**, not the per-event level.
Every video whose `Video_ID` contains the target class name (case
insensitive) is appended to the output CSV **twice**, with the features
transformed and the labels copied verbatim. Other videos pass through
unchanged.

```
Original CSV
  train_video_dribble_01.mp4         (no crossover in name → kept as-is)
  train_video_crossover_06_05_03.mp4 (qualifies)
  train_video_betweenLegs_02.mp4     (no match → kept as-is)

Output CSV
  train_video_dribble_01.mp4
  train_video_crossover_06_05_03.mp4
  train_video_crossover_06_05_03_mir.mp4   (mirror copy)
  train_video_crossover_06_05_03_aug.mp4   (mirror + noise + scale jitter)
  train_video_betweenLegs_02.mp4
```

### Why duplicate the whole video, not just the ±2 kernel frames

The GRU consumes 15-frame sliding windows ([60_build_gru_model.py:62-67](../60_build_gru_model.py#L62-L67))
grouped by `Video_ID`. The window with its right edge on the apex needs the
14 preceding frames — the dribble setup, the hand approach, the previous
bounce. Cropping a video to ±2 kernel frames would leave fewer than 15
frames in the new pseudo-clip and the window-builder would skip it. Keeping
the whole video preserves all valid windows that overlap the event.

Side effect: dribble-positive frames in those videos also get tripled.
Harmless — dribble is already abundant, and the crossover→dribble
propagation in [50_smoothing.md](50_smoothing.md#crossover--dribble-propagation)
already binds the two labels per frame.

## Selection rule

A video qualifies iff:

```python
class_name.lower() in video_id.lower()
```

For default `--class Crossover`, that means any `Video_ID` with "crossover"
anywhere in it. If your crossover clips aren't named that way, either
rename them or pass `--class` a substring that does appear.

The label column is **not** consulted for selection. This is deliberate:
your filenames are a manually curated signal of intent ("this clip was
recorded to showcase a crossover"), while the label column has soft kernel
values bleeding out from every nearby event. Filename matching is the
cleaner signal.

For `--class Hand_Touch`, the script looks for the literal substring
`hand_touch` (case insensitive). If your hand-touch clips are named
`handtouch_*.mp4` without the underscore, nothing will match and the
stats will report `0/N`.

## The three transforms

### Slot 1: horizontal mirror (deterministic)

Negate every `_X` column (`Rel_Ball_X`, `Rel_LeftElbow_X`, `Rel_RightElbow_X`,
`Rel_LeftWrist_X`, `Rel_RightWrist_X`, `Rel_LeftAnkle_X`, `Rel_RightAnkle_X`,
`Delta_Ball_X`) **and** swap every Left↔Right column pair:

```
Rel_LeftElbow_{X,Y}    ↔  Rel_RightElbow_{X,Y}
LeftElbow_Vis          ↔  RightElbow_Vis
Rel_LeftWrist_{X,Y,Z}  ↔  Rel_RightWrist_{X,Y,Z}
LeftWrist_Vis          ↔  RightWrist_Vis
Rel_LeftAnkle_{X,Y}    ↔  Rel_RightAnkle_{X,Y}
LeftAnkle_Vis          ↔  RightAnkle_Vis
Dist_Ball_L_Wrist      ↔  Dist_Ball_R_Wrist
Left_Wrist_Behind      ↔  Right_Wrist_Behind
```

The two operations together (negate-then-swap) produce a physically correct
horizontally-mirrored body. For Y / Z / Vis pairs, only the swap matters
(Y is unchanged by a horizontal flip; Z is the front-back axis, also
unchanged). For X pairs, the swap and the negation combine so the new
"left elbow X" equals `-(old right elbow X)`.

`Hands_Behind_Back_Count = Left_Wrist_Behind + Right_Wrist_Behind` is
invariant under the L↔R swap, so it's left untouched. `Ball_Detected` and
all label columns are also untouched.

### Slot 2: mirror + Gaussian noise + scale jitter

The mirror first (same as slot 1), then:

- **Noise** `~ N(0, σ)` added independently to every frame of every spatial
  coordinate (`Rel_*` and `Delta_Ball_*`). Default σ = 0.015 — small enough
  to not break event semantics, large enough to break exact memorization.
- **Scale jitter** by a single factor drawn from U[0.95, 1.05] per video,
  applied to every spatial column **including** `Norm_Torso_Height` and
  `Dist_Ball_*_Wrist`. Simulates camera-distance variation; using one
  factor per video keeps the scaling physically consistent across all
  joints within a clip.

Discrete columns are not touched by either step:

- `*_Vis` (MediaPipe visibility) — already noisy in source
- `Ball_Detected` (0/1 flag) — semantic, not geometric
- `Left/Right_Wrist_Behind`, `Hands_Behind_Back_Count` — discrete derived flags

Strictly, scaling `Rel_*Wrist_Z` would shift the wrist-behind threshold
crossings ([10_dataset_construction.md](10_dataset_construction.md#5-derived-features-post-hoc-in-inject_derived_features))
but we don't recompute the flags. The small (±5%) scale doesn't cross the
0.05 threshold for typical values, and the GRU is robust to occasional
threshold-edge inconsistency.

## NaN preservation

Frames where ball or pose detection failed have NaN in their feature
columns. All three transforms preserve NaN naturally: `NaN + x = NaN`,
`NaN * x = NaN`, and the swap copies the NaN to the partner column. The
downstream `fillna(0.0)` in `HoopsDataset.__init__` ([60_build_gru_model.py:44](../60_build_gru_model.py#L44))
runs as before.

## CLI

```
python 55_augment_class.py                          # Crossover, default σ and scale range
python 55_augment_class.py --class Dribble          # different target class
python 55_augment_class.py --sigma 0.02             # heavier noise
python 55_augment_class.py --scale-low 0.9 --scale-high 1.1
python 55_augment_class.py --seed 42                # different RNG draw
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--input` / `-i` | `../data/bskt/current_datasets/training_dataset_smooth_tracker_z.csv` | smoothed training CSV |
| `--output` / `-o` | `<input>_<class>_3x.csv` next to input | output path |
| `--class` / `-c` | `Crossover` | one of `Crossover`, `Dribble`, `Hand_Touch` |
| `--sigma` | `0.015` | Gaussian noise std-dev |
| `--scale-low` | `0.95` | lower bound of scale jitter |
| `--scale-high` | `1.05` | upper bound of scale jitter |
| `--seed` | `0` | NumPy RNG seed (matches the training script's `SEED`) |

The validation CSV is **never** touched — augmentation is a training-only
intervention. Validation must stay representative of real, unaugmented data
so F1 numbers compare to prior runs.

## Stats output

```
Loading .../training_dataset_smooth_tracker_z.csv...
  61,432 rows across 87 videos
  14/87 videos have 'crossover' in Video_ID
  output: 71,210 rows across 115 videos
  Crossover > 0 frames: 2,148 -> 6,444 (3.00x)
  Crossover apex (==1.0): 437 -> 1,311 (3.00x)
Writing .../training_dataset_smooth_tracker_z_crossover_3x.csv...
Done.
```

The 3.00x line is the verification: if it's not exactly 3.00, your
filename match is missing some videos, or some "crossover-containing"
videos have zero label positives in the CSV (rare — most likely they were
mislabeled or skipped in annotation).

## Plugging into training

[60_build_gru_model.py](../60_build_gru_model.py) reads its training CSV
from a hard-coded path inside `train_model()`. To consume the augmented
dataset, change that path to the new `_3x.csv` file:

```python
train_dataset = HoopsDataset("../data/bskt/current_datasets/training_dataset_smooth_tracker_z_crossover_3x.csv")
val_dataset   = HoopsDataset("../data/bskt/current_datasets/validation_dataset_smooth_tracker_z.csv")
```

Keep the val CSV pointing at the unaugmented file. Retrain with the same
`SEED = 0`, look at the **Crossover row** in the per-epoch log. Expect
recall to climb a few points, precision to maybe shed a point or two, and
the best-F1 checkpoint to land later in training (more data per epoch =
more steps to convergence).

## When this helps (and when it doesn't)

This pass amplifies the variety the model already sees in your crossover
clips. It can't manufacture diversity that isn't there. If your crossover
recordings all use the same player, same court, same camera angle, you'll
see saturation: more crossover examples but no improvement in val F1
beyond a point. The fix then is more raw clips, not more augmentation.

A reasonable iteration order:

1. Train baseline without augmentation. Note Crossover val F1.
2. Run this script with defaults. Retrain. Compare.
3. If Crossover val F1 improved but precision dropped meaningfully, raise
   the Crossover threshold in [60_build_gru_model.py](../60_build_gru_model.py)
   from `0.30` back toward `0.40` and re-measure event-count MAE in step 5.
4. If Crossover val F1 didn't move, the bottleneck is clip diversity,
   not example count — record more crossover variants instead of pushing
   σ higher.
