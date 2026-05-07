# Step 3 — Apply Soft-Label Smoothing

`50_apply_smoothing.py`

Converts the hard 0/1 ground-truth labels from step 2 into soft Gaussian-
shaped targets that train the GRU to produce well-calibrated peaks at the
exact frame of each event.

## Why smooth labels?

Hard 0/1 labels punish the model heavily for any miss, even by one frame.
A real basketball event (the moment the ball impacts the floor for a
dribble, the moment the ball changes hands for a crossover) is not a
discrete instant — annotation precision is roughly ±2 frames. Forcing the
model to predict 1.0 on the labeled frame and 0.0 on its neighbors
trains it to make sharp, brittle predictions that don't generalize.

A Gaussian-shaped soft target says: "the apex is here, the surrounding
frames are progressively *less* of an event." That gives the model:

- **More positive gradient signal** (5 frames per event contribute,
  not 1)
- **Calibrated probability output** — the model learns to produce
  shoulder probabilities of ~0.75 on near-event frames, exactly mirroring
  the soft label
- **Easier convergence** on the rare classes — Crossover apex frames are
  ~7% of training data; with smoothing, ~25% of frames get some positive
  signal

This makes downstream **event counting** (step 5) much more reliable
because the model produces clean peak-shaped probability traces that a
counter can convert into discrete event timestamps.

## The Gaussian kernel

```
          frame index
         ─2  ─1   0  +1  +2
         ──────────────────
target  0.25 0.75 1.0 0.75 0.25
```

5 frames wide. The center frame ("apex") gets 1.0; ±1 frames get 0.75;
±2 frames get 0.25; further frames stay 0. When two events are close
enough that their kernels overlap, the `max` of the two values is taken
at each frame — both peaks remain at 1.0, and the floor between them
rises proportionally.

## Algorithm

For each video and each label column (`Dribble`, `Crossover`):

### 1. Find continuous blocks of 1s

The hard labels from step 2 are typically multi-frame blocks (the
annotator held the tag down across several frames around the event).
Treat each contiguous run of 1s as one "block" representing one event.

```python
blocks = [(start, end), ...]
```

A block that runs to the last frame of the video is also captured.

### 2. Find the floor-impact apex within each block

For each block, the script picks the **single frame with the maximum
`Rel_Ball_Y`**. In screen coordinates Y increases downward, so max Y =
lowest position on screen = the moment the ball impacts the floor.

This is the key physical insight: a dribble's *event time* is the floor
impact, not the moment the ball leaves the hand or returns to the hand.
Floor impact gives a consistent, repeatable apex across recordings.

```python
impact_idx = block_start + np.argmax(block_y_coords)
new_labels[impact_idx] = 1.0
```

All other frames in the block (including the original 1s) become 0,
because the apex carries the entire event.

### 3. Apply the Gaussian kernel around each apex

```python
if new_labels[i] == 1.0:
    smoothed[i]   = 1.0
    smoothed[i-1] = max(smoothed[i-1], 0.75)
    smoothed[i+1] = max(smoothed[i+1], 0.75)
    smoothed[i-2] = max(smoothed[i-2], 0.25)
    smoothed[i+2] = max(smoothed[i+2], 0.25)
```

The `max(...)` ensures overlapping kernels (when two events are within 4
frames of each other) don't suppress each other.

## Crossover → Dribble propagation

After smoothing both columns independently, the script copies any
`Crossover > 0` values into `Dribble` where `Dribble == 0`:

```python
mask = (df['Crossover'] > 0) & (df['Dribble'] == 0)
df.loc[mask, 'Dribble'] = df.loc[mask, 'Crossover']
```

Why: a crossover is a **specific kind of dribble**. If the annotator
tagged a frame as "Crossover" but didn't also tag it as "Dribble"
(forgetting to add the Dribble tag is a common annotation mistake), the
frame should still be a positive dribble example. This step ensures the
hierarchy is consistent.

`Hand_Touch` is NOT smoothed — it's a sustained-contact label, not an
event-like apex, so the original 0/1 sequence is the correct target.

## Inputs and outputs

```
input:   ../data/bskt/training_dataset.csv          (hard labels)
output:  ../data/bskt/training_dataset_smooth.csv   (soft labels)
```

The script's `__main__` block has both training and validation calls; one
is commented out. Run with whichever path you need.

## What the data looks like after smoothing

A frame's `Crossover` value is now one of `{0.0, 0.25, 0.75, 1.0}` (and
occasionally `0.5` from arithmetic on overlapping kernels — uncommon).
Same for `Dribble`.

The training script's `HoopsDataset` reads these float values directly
as targets — `nn.BCELoss` accepts soft targets in `[0, 1]` natively.
This is what makes the model produce smooth peak-shaped probability
output instead of step-function output, which is what makes event
counting work.

## Pitfall: smoothing widens what counts as "event present"

When you measure metrics on smoothed labels, a frame at 0.25 is still
technically "above zero." The training script handles this by computing
two F1 metrics:

- **Inclusive F1** (label ≥ 0.5): treats apex + ±1 shoulders as positive,
  ±2 shoulders as negative. Tolerant.
- **Apex F1** (label == 1.0): only the apex frame is positive. Strict.

The eval script (step 5) measures count accuracy at the event level,
which is how you should evaluate end-to-end performance. F1 is just a
training-loop diagnostic.
