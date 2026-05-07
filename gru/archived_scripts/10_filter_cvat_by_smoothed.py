import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

CSV_FILES = [
    "../data/bskt/training_dataset_smooth.csv",
    "../data/bskt/validation_dataset_smooth.csv",
]
CVAT_FOLDER = "../data/bskt/cvat_exports"
FILTER_LABELS = ("Dribble", "Crossover")
OUTPUT_SUFFIX = "_filtered.xml"


def build_lookup(csv_paths):
    """Return {video_id: {frame_id: {'Dribble': v, 'Crossover': v}}}."""
    frames = {}
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  (skip) CSV not found: {path}")
            continue
        print(f"  Loading {path}")
        df = pd.read_csv(path, usecols=["Video_ID", "Frame_ID", "Dribble", "Crossover"])
        for row in df.itertuples(index=False):
            frames.setdefault(row.Video_ID, {})[int(row.Frame_ID)] = {
                "Dribble": float(row.Dribble),
                "Crossover": float(row.Crossover),
            }
    return frames


def xml_path_to_video_id(xml_path):
    """test_dribble_cvat_exported.xml -> test_dribble.mp4"""
    base = os.path.basename(xml_path)
    stem = base[: -len("_cvat_exported.xml")]
    return stem + ".mp4"


def filter_xml(xml_path, video_frames):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    removed = 0
    kept = 0
    missing_frames = 0

    for image in root.findall("image"):
        try:
            frame_id = int(image.attrib.get("id"))
        except (TypeError, ValueError):
            continue

        csv_row = video_frames.get(frame_id)

        for tag in list(image.findall("tag")):
            label = tag.attrib.get("label")
            if label not in FILTER_LABELS:
                continue
            if csv_row is None:
                image.remove(tag)
                removed += 1
                missing_frames += 1
                continue
            if csv_row.get(label, 0.0) == 1.0:
                kept += 1
            else:
                image.remove(tag)
                removed += 1

    out_path = xml_path[: -len(".xml")] + OUTPUT_SUFFIX
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return kept, removed, missing_frames, out_path


def main():
    print("Loading smoothed datasets...")
    lookup = build_lookup(CSV_FILES)
    print(f"Indexed {len(lookup)} videos.\n")

    pattern = os.path.join(CVAT_FOLDER, "*_exported.xml")
    xml_files = sorted(glob.glob(pattern))
    if not xml_files:
        print(f"No *_exported.xml files found in {CVAT_FOLDER}")
        return

    print(f"Found {len(xml_files)} exported XML files. Filtering Dribble/Crossover tags...")
    print("-" * 60)

    total_kept = total_removed = total_missing = 0
    for xml_path in xml_files:
        video_id = xml_path_to_video_id(xml_path)
        video_frames = lookup.get(video_id)
        if video_frames is None:
            print(f"[SKIP] {os.path.basename(xml_path)} -> no CSV match for '{video_id}'")
            continue

        kept, removed, missing, out_path = filter_xml(xml_path, video_frames)
        total_kept += kept
        total_removed += removed
        total_missing += missing
        print(
            f"[{video_id}] kept={kept:4d}  removed={removed:4d}  "
            f"missing_frames={missing:3d}  -> {os.path.basename(out_path)}"
        )

    print("-" * 60)
    print(f"Totals: kept={total_kept}  removed={total_removed}  missing_frames={total_missing}")


if __name__ == "__main__":
    main()
