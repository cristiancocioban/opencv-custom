# GRU Action Recognition Pipeline

End-to-end pipeline for detecting and counting basketball ball-handling events
(dribbles, crossovers, hand-touches) from monocular RGB video.

## Pipeline overview

```
   raw .mp4 videos
         │
         ▼
[10] build_dribble_dataset.py     ──►  training_data.csv
         │                              (features per frame, labels = 0)
         │
         ▼
[CVAT manual annotation]          ──►  *_cvat_final.xml
         │                              (ground-truth event tags per frame)
         │
         ▼
[40] merge_ground_truth.py        ──►  training_dataset.csv
         │                              (hard 0/1 labels merged in)
         │
         ▼
[50] apply_smoothing.py           ──►  training_dataset_smooth.csv
         │                              (Gaussian-shaped soft labels)
         │
         ▼
[60] build_gru_model.py           ──►  hoops_world_model_best.pth
         │                              hoops_world_model_best_f1.pth
         │
         ├──►  [70] evaluate_event_counts.py    (offline batch eval)
         │
         └──►  [90] test_inference.py           (live overlay on a video)
```

## Per-step documentation

| Step | Script | Doc | What it does |
|------|--------|-----|--------------|
| 1 | `10_build_dribble_dataset.py` | [10_dataset_construction.md](10_dataset_construction.md) | Extract ball + pose features per frame |
| 2 | `40_merge_ground_truth.py` | [40_merge_ground_truth.md](40_merge_ground_truth.md) | Overlay manual CVAT labels onto features |
| 3 | `50_apply_smoothing.py` | [50_smoothing.md](50_smoothing.md) | Convert hard labels into Gaussian soft targets |
| 4 | `60_build_gru_model.py` | [60_model_training.md](60_model_training.md) | Train the multi-head GRU |
| 5 | `70_evaluate_event_counts.py` | [70_evaluation.md](70_evaluation.md) | Measure event-count accuracy |

## Why this pipeline

The end goal is **event counting**: how many dribbles and crossovers happened
in a video. That goal shapes every choice upstream:

- **Frame-level features** (ball position relative to hip, joint visibilities,
  ball velocity) are general enough to capture many ball-handling moves
  beyond just the three currently labeled.
- **Soft Gaussian labels** train the model to produce a *peak-shaped*
  probability around each event, rather than a binary 0→1 step. Peak shapes
  are easier for a counter to convert into discrete events.
- **Two complementary counters** (hysteresis state machine and peak detection)
  let us validate that the model's probability output is genuinely
  event-shaped and not relying on counter-specific tricks.
- **Apex F1 + inclusive F1 + event-count MAE** give three diagnostic angles:
  apex F1 measures peak fidelity, inclusive F1 measures temporal envelope
  coverage, MAE measures the production metric.

## Pre-existing assumptions

- Frame rate is treated as constant per video (typically 30fps). Frame-based
  thresholds (`min_gap=5`, `crossover_window=5`) are tuned at that rate. If
  you process video at substantially different fps, scale them.
- Pose comes from MediaPipe Pose at `model_complexity=2` (the heaviest
  variant). A lighter pose model would shift the per-frame features and
  require retraining.
- Ball detection comes from a project-specific YOLO model
  (`ball_detection_v26n_*.pt` / `_raw.onnx`) that emits class 0 = ball. Any
  swap to a different ball detector requires retraining the GRU because
  detection noise patterns are baked into the learned features.

## Where outputs live

```
data/bskt/
  training_data.csv                         ← step 1 output (features only)
  training_dataset.csv                      ← step 2 output (with hard labels)
  training_dataset_smooth.csv               ← step 3 output (with soft labels)
  current_datasets/
    training_dataset_smooth_tracker.csv     ← active training input for step 4
    validation_dataset_smooth_tracker.csv   ← active validation input
models/
  hoops_world_model_best.pth                ← step 4 output: lowest val loss
  hoops_world_model_best_f1.pth             ← step 4 output: highest val F1
```

The `_tracker` suffix indicates the dataset was built with `cv2.BallTracker`
(YOLO + TrackerNano) rather than stand-alone YOLO. Tracker datasets have
fewer ball-detection gaps and produce noticeably better-trained models.
