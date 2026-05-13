"""
Oversample one action class in a smoothed training CSV by adding mirrored and
mirrored+noise+scale-jittered copies of every video whose filename contains
the class name (case insensitive). Default class is Crossover, so any
Video_ID with "crossover" anywhere in its name qualifies.

Each qualifying video becomes 3 entries in the output CSV:
    1. the original rows                                    (Video_ID = X.mp4)
    2. a horizontal-mirror copy                             (Video_ID = X_mir.mp4)
    3. a horizontal-mirror + Gaussian-noise + scale-jitter  (Video_ID = X_aug.mp4)

Videos whose name doesn't contain the class string are passed through
unchanged.

Mirror: negates every _X feature and swaps Left<->Right column pairs (elbow /
wrist / ankle / Dist_Ball_*_Wrist / Left_Wrist_Behind). Y, Z, and Vis values
are swapped but not negated. Hands_Behind_Back_Count and Ball_Detected are
left alone (the count is L+R, invariant under L<->R swap).

Gaussian noise (sigma 0.015 default) on Rel_* coordinates and Delta_Ball_*.

Scale jitter: a single factor drawn from U[0.95, 1.05] per augmented video,
applied to every spatial column (all Rel_*, Delta_Ball_*, Dist_Ball_*_Wrist,
Norm_Torso_Height).

Discrete columns (*_Vis, Ball_Detected, Left/Right_Wrist_Behind,
Hands_Behind_Back_Count) are not noised or scaled. Label columns (Dribble,
Crossover, Hand_Touch) are copied verbatim, so smoothed Gaussian targets are
preserved.

Run:
    python 55_augment_class.py
    python 55_augment_class.py --class Crossover
    python 55_augment_class.py --class Dribble --sigma 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "data" / "bskt" / "current_datasets"
    / "training_dataset_smooth_tracker_z.csv"
)


X_FEATURES = [
    "Rel_Ball_X",
    "Rel_LeftElbow_X", "Rel_RightElbow_X",
    "Rel_LeftWrist_X", "Rel_RightWrist_X",
    "Rel_LeftAnkle_X", "Rel_RightAnkle_X",
    "Delta_Ball_X",
]

LR_PAIRS = [
    ("Rel_LeftElbow_X", "Rel_RightElbow_X"),
    ("Rel_LeftElbow_Y", "Rel_RightElbow_Y"),
    ("LeftElbow_Vis", "RightElbow_Vis"),
    ("Rel_LeftWrist_X", "Rel_RightWrist_X"),
    ("Rel_LeftWrist_Y", "Rel_RightWrist_Y"),
    ("Rel_LeftWrist_Z", "Rel_RightWrist_Z"),
    ("LeftWrist_Vis", "RightWrist_Vis"),
    ("Rel_LeftAnkle_X", "Rel_RightAnkle_X"),
    ("Rel_LeftAnkle_Y", "Rel_RightAnkle_Y"),
    ("LeftAnkle_Vis", "RightAnkle_Vis"),
    ("Dist_Ball_L_Wrist", "Dist_Ball_R_Wrist"),
    ("Left_Wrist_Behind", "Right_Wrist_Behind"),
]

NOISE_COLS = [
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y",
    "Rel_RightElbow_X", "Rel_RightElbow_Y",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "Rel_LeftWrist_Z",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "Rel_RightWrist_Z",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y",
    "Delta_Ball_X", "Delta_Ball_Y",
]

SCALE_COLS = [
    "Rel_Ball_X", "Rel_Ball_Y",
    "Rel_LeftElbow_X", "Rel_LeftElbow_Y",
    "Rel_RightElbow_X", "Rel_RightElbow_Y",
    "Rel_LeftWrist_X", "Rel_LeftWrist_Y", "Rel_LeftWrist_Z",
    "Rel_RightWrist_X", "Rel_RightWrist_Y", "Rel_RightWrist_Z",
    "Rel_LeftAnkle_X", "Rel_LeftAnkle_Y",
    "Rel_RightAnkle_X", "Rel_RightAnkle_Y",
    "Delta_Ball_X", "Delta_Ball_Y",
    "Dist_Ball_L_Wrist", "Dist_Ball_R_Wrist",
    "Norm_Torso_Height",
]


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in X_FEATURES:
        out[col] = -out[col]
    for left, right in LR_PAIRS:
        out[left], out[right] = out[right].copy(), out[left].copy()
    return out


def add_noise(df: pd.DataFrame, sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    for col in NOISE_COLS:
        # NaN + value = NaN, so missing-frame rows stay missing.
        out[col] = out[col] + rng.normal(0.0, sigma, size=n)
    return out


def scale_jitter(
    df: pd.DataFrame, low: float, high: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = df.copy()
    factor = float(rng.uniform(low, high))
    for col in SCALE_COLS:
        out[col] = out[col] * factor
    return out


def _split_video_id(vid: str) -> tuple[str, str]:
    if vid.endswith(".mp4"):
        return vid[:-4], ".mp4"
    return vid, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triple a target class in a smoothed training CSV.",
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help="path to the smoothed training CSV",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="output CSV path (default: <input_stem>_<class>_3x.csv next to input)",
    )
    parser.add_argument(
        "--class", "-c", dest="cls", default="Crossover",
        choices=["Crossover", "Dribble", "Hand_Touch"],
        help="action class whose videos get tripled (default: Crossover). "
             "A video qualifies iff its Video_ID contains the class name "
             "(case insensitive).",
    )
    parser.add_argument("--sigma", type=float, default=0.015,
                        help="Gaussian noise std-dev for spatial coords")
    parser.add_argument("--scale-low", type=float, default=0.95)
    parser.add_argument("--scale-high", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    in_path: Path = args.input
    out_path: Path = (
        args.output
        if args.output is not None
        else in_path.with_name(f"{in_path.stem}_{args.cls.lower()}_3x.csv")
    )

    print(f"Loading {in_path}...")
    df = pd.read_csv(in_path)
    n_videos = df["Video_ID"].nunique()
    print(f"  {len(df):,} rows across {n_videos} videos")

    needle = args.cls.lower()
    target_videos = {
        vid for vid in df["Video_ID"].unique() if needle in str(vid).lower()
    }
    print(
        f"  {len(target_videos)}/{n_videos} videos have '{needle}' in Video_ID"
    )

    chunks: list[pd.DataFrame] = [df]

    for vid, group in df.groupby("Video_ID", sort=False):
        if vid not in target_videos:
            continue
        stem, ext = _split_video_id(vid)

        mirrored = mirror(group)
        mirrored["Video_ID"] = f"{stem}_mir{ext}"
        chunks.append(mirrored)

        jittered = scale_jitter(
            add_noise(mirror(group), args.sigma, rng),
            args.scale_low, args.scale_high, rng,
        )
        jittered["Video_ID"] = f"{stem}_aug{ext}"
        chunks.append(jittered)

    out = pd.concat(chunks, ignore_index=True)

    pos_before = int((df[args.cls] > 0).sum())
    pos_after = int((out[args.cls] > 0).sum())
    apex_before = int((df[args.cls] == 1.0).sum())
    apex_after = int((out[args.cls] == 1.0).sum())

    print(f"  output: {len(out):,} rows across {out['Video_ID'].nunique()} videos")
    print(
        f"  {args.cls} > 0 frames: {pos_before:,} -> {pos_after:,} "
        f"({pos_after / max(pos_before, 1):.2f}x)"
    )
    print(
        f"  {args.cls} apex (==1.0): {apex_before:,} -> {apex_after:,} "
        f"({apex_after / max(apex_before, 1):.2f}x)"
    )

    print(f"Writing {out_path}...")
    out.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
