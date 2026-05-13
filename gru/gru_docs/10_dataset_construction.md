# Step 1 — Dataset Construction

`10_build_dribble_dataset.py`

Turns a folder of raw `.mp4` clips into a single CSV where each row is one
frame and each column is a model-ready feature.

## Inputs

- A directory of basketball video clips (`../raw_dribbling_videos/*.mp4` by
  default) — one player per video, no scene cuts within a clip.
- A YOLO ball-detection model (`--weights`). Class 0 must be "ball."
- (Optional) `cv2.BallTracker` configuration: nano backbone + neckhead
  ONNX files. Selected with `--tracker`. Recommended over stand-alone YOLO
  because it bridges occlusions and motion blur far better.
- (Optional) An already-trained GRU checkpoint (`--gru-weights`). Used to
  auto-fill the action labels at the end. Skip with `--no-gru` when
  generating fresh CSVs to train a new GRU.

## Per-frame processing

For every frame of every video, the script does five things in order:

### 1. Ball detection

Two backends, picked at the command line:

- **Stand-alone YOLO** (`detect_ball`): runs the detector on the full frame,
  picks the highest-confidence class-0 box, returns its center normalized
  to `[0, 1]` of the frame dimensions. Returns `None` if no detection.
- **`cv2.BallTracker`** (`detect_ball_with_tracker`, used when `--tracker`):
  runs the project's custom tracker (YOLO seed + TrackerNano template
  match). Maintains internal state across frames so brief occlusions are
  bridged. Returns `normBbox` directly. Recommended.

A new tracker is created **per video** (`create_ball_tracker` is called
inside the loop) so calibration / color model / drift-recovery state
restart for each clip.

### 2. Pose extraction

MediaPipe Pose at `model_complexity=2` (the heaviest, most accurate
variant), `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`.
Returns 33 landmarks; we use 10 of them.

### 3. Hip-center anchor

```python
hip_center = avg_xy(left_hip, right_hip)
```

The hip mid-point becomes the **origin** for every subsequent coordinate.
This is the single most important normalization decision in the pipeline:
it eliminates global player position from the features. A dribble looks the
same to the model whether the player is at the left or right of the frame.

If the hip can't be located (no pose, or one hip is missing), the row's
landmark-derived features stay `NaN` and are filled with `0.0` later.

### 4. Feature computation

For each tracked landmark (left/right elbow, ankle), three values
are stored: `(x - hip_x, y - hip_y, visibility)`. For each wrist, a fourth
value is stored — `Rel_{Left,Right}Wrist_Z` — pulled directly from
MediaPipe. MediaPipe's `z` is already expressed **relative to the hip
mid-point** (negative in front of the camera, positive behind), so it does
NOT need to be re-anchored against `hip_center` the way `x` and `y` are.

The visibility comes straight from MediaPipe — we deliberately do NOT
threshold it, so the GRU learns to trust or distrust each landmark based
on its raw confidence.

The ball center is also stored relative to the hip (`Rel_Ball_X`,
`Rel_Ball_Y`).

`Norm_Torso_Height` is `|shoulder_center_y − hip_y|`. Acts as an implicit
**scale normalizer**: distances and velocities can be interpreted relative
to the player's torso length, which gives the model a frame-rate- and
camera-distance-tolerant scale.

### 5. Derived features (post-hoc, in `inject_derived_features`)

After all per-frame rows are gathered, the following extra features are
computed from the time-series:

- `Dist_Ball_L_Wrist`, `Dist_Ball_R_Wrist`: Euclidean distance from ball to
  each wrist (in the hip-relative frame). Strong signal for `Hand_Touch`.
- `Delta_Ball_X`, `Delta_Ball_Y`: per-frame ball velocity, computed from a
  forward-filled ball position. **Forward-fill is intentional**: a missed
  detection produces zero velocity ("we assume the ball didn't move")
  rather than a NaN that gets silently mapped to 0. The GRU sees
  `Ball_Detected` separately, so it can learn to distrust velocity values
  that came from filled frames.
- `Left_Wrist_Behind`, `Right_Wrist_Behind`: discrete indicators set to
  `1.0` when the corresponding wrist's `Rel_Wrist_Z` exceeds `+0.05`
  (i.e. the wrist has crossed behind the hip plane), `0.0` otherwise. The
  `0.05` threshold ignores noise around `z = 0` from imperfect pose
  tracking. Designed as the primary cue for **Behind-the-Back** dribbles.
- `Hands_Behind_Back_Count`: sum of the two indicators above (`0`, `1`,
  or `2`). A two-hand crossover **in front** of the body should keep this
  at `0`; a Behind-the-Back move briefly drives it to `1` (and rarely `2`)
  as the dribbling hand sweeps behind the torso. Combined with the
  Between-the-Legs cue (low `Rel_Ball_Y` with both wrists in front), this
  pair lets the GRU disambiguate the two trick-dribble classes.

### 6. The Ball_Detected flag

`row["Ball_Detected"] = 1.0 if ball_norm else 0.0`

Critical bit of metadata. Without it, a frame where YOLO missed the ball
(stored as `Rel_Ball_X = 0, Rel_Ball_Y = 0` after fillna) would look
identical to a frame where the ball *really is* at the hip center. This
ambiguity poisons the gradient. The flag lets the GRU treat (0, 0) at
`Ball_Detected = 0` differently from (0, 0) at `Ball_Detected = 1`.

## The 31 features (final feature vector)

Order matters and must be identical across training, evaluation, and
inference. The original 25 features (now reshuffled within the table)
were the body-pose + ball-position + 2D-velocity set; `Ball_Detected`
was added later to fix the missed-detection ambiguity; the wrist-depth
columns (`Rel_LeftWrist_Z`, `Rel_RightWrist_Z`, `Left_Wrist_Behind`,
`Right_Wrist_Behind`, `Hands_Behind_Back_Count`) were added when
Between-the-Legs and Behind-the-Back recognition was introduced.

```
 0  Rel_Ball_X                ball x relative to hip (normalized)
 1  Rel_Ball_Y                ball y relative to hip
 2  Rel_LeftElbow_X
 3  Rel_LeftElbow_Y
 4  LeftElbow_Vis             MediaPipe visibility, raw
 5  Rel_RightElbow_X
 6  Rel_RightElbow_Y
 7  RightElbow_Vis
 8  Rel_LeftWrist_X
 9  Rel_LeftWrist_Y
10  Rel_LeftWrist_Z           MediaPipe z, hip-relative (+ = behind body)
11  LeftWrist_Vis
12  Rel_RightWrist_X
13  Rel_RightWrist_Y
14  Rel_RightWrist_Z          MediaPipe z, hip-relative (+ = behind body)
15  RightWrist_Vis
16  Rel_LeftAnkle_X
17  Rel_LeftAnkle_Y
18  LeftAnkle_Vis
19  Rel_RightAnkle_X
20  Rel_RightAnkle_Y
21  RightAnkle_Vis
22  Norm_Torso_Height         implicit scale normalizer
23  Dist_Ball_L_Wrist
24  Dist_Ball_R_Wrist
25  Delta_Ball_Y              forward-filled velocity
26  Delta_Ball_X
27  Left_Wrist_Behind         1.0 if Rel_LeftWrist_Z > 0.05
28  Right_Wrist_Behind        1.0 if Rel_RightWrist_Z > 0.05
29  Hands_Behind_Back_Count   sum of the two indicators (0/1/2)
30  Ball_Detected             1.0 if YOLO/tracker found ball this frame
```

If you ever reorder, add, or remove features, you MUST regenerate the
training data, retrain the GRU, and update both `60_build_gru_model.py`
(the `feature_cols` list inside `HoopsDataset`) and `90_test_inference.py`
(its inline feature builder).

## The CSV output

Default path: `../data/bskt/training_data.csv`. Columns:

- `Video_ID`, `Frame_ID` — composite key, used to split data and build
  sliding windows that don't cross video boundaries.
- All 26 features above.
- `Dribble`, `Crossover`, `Hand_Touch` — action labels. Filled by either:
  - the optional GRU auto-labeling step (when an existing model is passed
    via `--gru-weights`), or
  - left as `0` if `--no-gru` (the case when bootstrapping a new model
    from scratch — labels will be supplied later by manual annotation
    in step 2).

## When to use --no-gru

Set `--no-gru` whenever the labels in the existing checkpoint cannot be
trusted for your new clips — e.g., when you have new players, new camera
angles, or you're starting from a clean slate. The auto-label is only safe
to use as a *seed* for re-annotation in CVAT, never as final ground truth.

## Debug overlay

`--debug` opens a `cv2.imshow` window showing the hip anchor (yellow dot),
ball detection (red circle), and the line connecting them. Useful for
sanity-checking the YOLO confidence threshold and the pose tracking
quality on a few frames. Press `q` to abort.
