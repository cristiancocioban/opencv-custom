# Step 2 — Merge Ground Truth

`40_merge_ground_truth.py`

Overlays manual event annotations (created in CVAT) onto the per-frame
feature CSV produced by step 1.

## Why this step exists

Step 1 produces features but no trustworthy labels. The optional GRU
auto-label (when a previous-generation model is loaded) is only ever used
as a *seed* to make CVAT pre-tagging easier — never as final ground truth.

Step 2 is where humans correct that seed and the corrected labels
overwrite whatever was in the feature CSV.

## Workflow around this script

```
   training_data.csv          ←── output of step 1 (features + seed labels)
        │
        ▼
[20] csv_to_cvat.py           converts the seed labels into CVAT XML so the
        │                     annotator only fixes mistakes, doesn't start blank
        ▼
   CVAT (manual annotation)   annotator confirms / adds / removes Dribble,
        │                     Crossover, Hand_Touch tags per frame
        ▼
   *_cvat_final.xml           one XML per video
        │
        ▼
[40] merge_ground_truth.py    ← this script
        │
        ▼
   training_dataset.csv       (features + corrected hard labels)
```

## What this script does

1. Loads the master CSV from step 1 (`../data/bskt/training_data.csv` by
   default).
2. **Resets** `Dribble`, `Crossover`, `Hand_Touch` columns to `0` for
   every row. This is intentional — we don't want any leftover seed
   labels that the annotator did not explicitly confirm. Only what the
   annotator marked in CVAT will end up positive.
3. For each entry in the `corrected_files` dict (keyed by video filename
   in the CSV, valued by path to that video's CVAT XML):
   - Skips the entry if the XML doesn't exist (with a warning print)
   - Parses the XML; CVAT stores per-frame annotations as
     `<image id="N">` elements containing `<tag label="...">` children.
   - For each tag, finds the row in the DataFrame matching
     `Video_ID == video_filename` and `Frame_ID == N`, and sets the
     corresponding label column to `1`.
4. Writes the merged result to `../data/bskt/training_dataset.csv`.

## The `corrected_files` registry

```python
corrected_files = {
    'crossover_02_05_1.mp4': '../data/bskt/cvat_exports/crossover_02_05_1_cvat_final.xml',
    # Add more videos here as they get annotated...
}
```

This dict is the **only place** the human-annotated truth meets the
features. Adding a newly-annotated video means:
1. Drop the CVAT-exported XML in `../data/bskt/cvat_exports/`.
2. Add one entry to `corrected_files`.
3. Rerun the script.

Videos in the CSV that are *not* in `corrected_files` keep their reset-to-
zero labels — they will train as if no event ever happened, which is
clearly wrong. So either annotate every clip you want to use, or filter
the CSV to drop unlabeled videos before training.

## CVAT label conventions

The script understands exactly three CVAT tag labels: `Dribble`,
`Crossover`, `Hand_Touch`. Tags with any other label name are silently
ignored (no warning). If you add a new event class, this script needs an
extra branch in the tag-handling block, plus a new column in the CSV
schema, plus a corresponding output in the GRU classifier head.

## Output

Default path: `../data/bskt/training_dataset.csv`. Same schema as the
input CSV but with the three label columns now containing `0` or `1`
based on the merged ground truth.

These are still **hard** labels (single-frame 0/1 markers). Step 3
(smoothing) converts them into soft Gaussian targets for training.

## Common pitfall

If the `Frame_ID` numbering in the CVAT XML doesn't match the `Frame_ID`
in the CSV, no labels will land. This happens if CVAT was given a video
whose first frame is offset (e.g., a re-encoded clip with a leading black
frame). Spot-check by comparing one or two known events: open the source
video at, say, frame 142 (a labeled crossover); confirm the CSV row
`Video_ID == X, Frame_ID == 142` now has `Crossover == 1` after running
this script. If the alignment is off by a constant, find and remove the
re-encoding step before re-annotating.
