# Step 4 — Train the GRU World Model

`60_build_gru_model.py`

Trains a multi-head GRU that, given the last 15 frames of features,
predicts both:

1. The probability that each of `Dribble`, `Crossover`, `Hand_Touch` is
   happening at the **last frame** of the window (classifier head).
2. The position of the ball at the **next** frame (predictor head, used
   downstream as a sanity / "AI target" overlay).

## Inputs

- `training_dataset_smooth_tracker.csv` (output of step 3 with the active
  tracker-based dataset)
- `validation_dataset_smooth_tracker.csv` (held-out for early stopping
  and metric tracking)

Both CSVs must contain the same 26 feature columns and the three label
columns (`Dribble`, `Crossover`, `Hand_Touch`) as soft Gaussian targets.

## Architecture

```
Input (15 frames × 26 features)
      │
      ▼
GRU layer 1  (hidden=64)        ─┐
GRU layer 2  (hidden=64)        ─┘  dropout=0.2 between layers
      │
   take last frame's hidden state h_15
      │
      ├──► classifier_head ──► σ(3)  → Dribble, Crossover, Hand_Touch
      │      Linear(64, 32)
      │      ReLU
      │      Dropout(0.3)            ← bumped from 0.2 to 0.3 to tighten
      │      Linear(32, 3)             cross-seed variance
      │      Sigmoid
      │
      └──► predictor_head ──► (2)    → next_ball_X, next_ball_Y
             Linear(64, 32)
             ReLU
             Dropout(0.2)
             Linear(32, 2)
             (no activation — coords are real-valued)
```

The GRU is bidirectional in spirit only — `nn.GRU` with `num_layers=2`
stacks two layers, both processing in temporal order. The window-level
prediction uses **only the last hidden state**, so the model is summarizing
what it saw over 15 frames into one prediction at the right edge.

The classifier head and predictor head share the GRU trunk. Loss is the
sum of action and coord losses, with the coord loss scaled `× 10` to
balance their gradient magnitudes (BCE is naturally larger than MSE on
normalized coords).

## Loss functions

```python
action_criterion = nn.BCELoss(reduction='none')   # per-class loss
coord_criterion  = nn.MSELoss()
total_loss = action_loss.mean() + 10 * coord_loss
```

`BCELoss` natively supports soft targets — `BCE(p=0.7, t=0.75)` is a
well-defined regression-flavored loss. That's why step 3's Gaussian
smoothing works out of the box without changing the loss function.

## Training configuration

```python
optimizer:    Adam(lr=0.001, weight_decay=1e-4)
batch_size:   32
window_size:  15
num_epochs:   60
patience:     12 (epochs without F1 improvement before early stopping)
SEED:         0  (forces all RNG sources for reproducibility)
```

`weight_decay=1e-4` is the L2 regularization. Together with the head
dropouts and the GRU's internal `dropout=0.2`, this is the regularization
stack that keeps cross-seed F1 variance manageable.

## Reproducibility

Five RNGs are seeded at the top of `train_model()`:

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

This is essential — without it, two consecutive runs can produce val F1
that differs by 8+ points (we measured 0.84 vs 0.93 on different seed
luck before locking the seed). Fixed seed = any change in numbers is
real signal, not init noise.

## Metrics tracked per epoch

For each of the three classes, the training loop reports four numbers
per epoch on both train and val:

### Loss (BCE)

The actual training signal. Drops monotonically when the model is
learning; the gap between train and val loss is the conventional
overfitting signal — but be careful, the gap reflects **calibration**
(confidence) at least as much as accuracy. A correct prediction at p=0.99
costs ~0.01; the same correct prediction at p=0.7 costs ~0.36. Big train/
val loss gap can be confidence collapse on training, not bad val accuracy.

### P / R / F1 (inclusive — labels ≥ 0.5)

Predictions thresholded at 0.5; ground-truth thresholded at 0.5 (so apex
+ ±1 shoulder frames count as positive, ±2 shoulders count as negative).
This is the temporal-envelope metric: how well does the model's positive
prediction period overlap with the labeled positive period?

### P / R / F1 (apex — labels == 1.0)

Same predictions thresholded at 0.5; ground truth requires `label == 1.0`
(only the peak frame is positive, only frames at exactly `0.0` are
negative, shoulders excluded entirely from the metric). This is the peak-
fidelity metric: how reliably does the model fire on the actual apex?

For event counting (the real goal), apex F1 is the more relevant signal
because counting cares about producing **one sharp prediction per event**.

### Per-class breakdown

The classifier output is a 3-vector. Each of Dribble / Crossover /
Hand_Touch gets its own loss, P/R/F1, and per-epoch trajectory. Crossover
typically lags the other two because it's the rarest class (~7% positives
in training).

## Early stopping & checkpointing

Two checkpoints are saved during training, updated whenever a new best is
seen:

| File | Saves on | Use for |
|------|----------|---------|
| `hoops_world_model_best.pth` | lowest val loss | well-calibrated probabilities |
| `hoops_world_model_best_f1.pth` | highest val macro F1 | event counting / inference |

Early stopping fires when **val macro F1** has not improved for `patience`
epochs. F1 was chosen instead of val loss because we observed loss can
hit fluke lows on early epochs (when the model is uniformly uncertain →
low BCE) and trigger a premature stop while F1 is still climbing.

## What the training output looks like

```
Epoch [35/60] | Train Loss: 0.1797 | Val Loss: 0.2945 | Val macro F1: 0.9188 | Apex: 0.92
  Coord  -> Train: 0.0023 | Val: 0.0016
  Dribble    -> Train: 0.2143 | Val: 0.2882
                Train P/R/F1: 0.92/0.92/0.92  Val P/R/F1: 0.97/0.88/0.93
                Apex T/V F1: 0.95/0.93
  Crossover  -> Train: 0.0980 | Val: 0.2388
                Train P/R/F1: 0.91/0.84/0.87  Val P/R/F1: 0.94/0.91/0.92
                Apex T/V F1: 0.93/0.92
  Hand_Touch -> Train: 0.1562 | Val: 0.3073
                Train P/R/F1: 0.94/0.96/0.95  Val P/R/F1: 0.90/0.92/0.91
                Apex T/V F1: 0.96/0.91
  --> New best val loss (0.2945) -> hoops_world_model_best.pth
  --> New best macro F1 (0.9188) -> hoops_world_model_best_f1.pth
```

End-of-run summary:

```
Training Complete!
  Best val loss: 0.2901 at epoch 38 -> hoops_world_model_best.pth
  Best macro F1: 0.9188 at epoch 35 -> hoops_world_model_best_f1.pth
```

## Iteration tips

- **For ranking experiments:** keep `SEED = 0`, change one knob, compare
  best F1. Treat any difference under ~1 F1 point as noise.
- **For variance estimates:** train the same config across `SEED ∈
  {0, 1, 7, 42, 100}`. Spread of more than ~3 F1 points means
  regularization is too loose — bump classifier dropout or weight decay.
- **For event counting:** train, then run step 5 (`70_evaluate_event_
  counts.py`). Frame-level F1 is a proxy; event-count MAE is the truth.
- **If training is hitting `num_epochs=60` without early-stopping:** the
  model is still improving. Bump to 80 or 100 and re-run. If the curve
  has actually flattened, early stopping will fire.
