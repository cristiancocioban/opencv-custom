"""
ball_tracker.py - Basketball tracking using YOLO + OpenCV TrackerVit / TrackerNano

Two-phase tracking approach:
  1. YOLO detects the basketball every N frames or when tracking is lost
  2. OpenCV tracker (TrackerVit or TrackerNano) handles continuous frame-to-frame tracking

Template bank:
  Stores ball crops from confident detections at different distances and under
  different lighting conditions. New YOLO detections are validated against the
  bank via HSV histogram comparison before being accepted, which filters out
  false positives (e.g. a jersey number that looks like a ball to YOLO).
  High-confidence tracker frames also feed the bank periodically so it stays
  up-to-date as the ball moves closer/further from the camera.

Color coding:
  GREEN bounding box = OpenCV tracker is tracking
  RED bounding box   = YOLO detected the ball
  ORANGE bounding box = YOLO detected but failed template validation (rejected)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

class TemplateBank:
    """
    Fixed-size store of ball image crops used to validate new YOLO detections.

    Why this helps:
    - As the player moves around the court the ball appears at different sizes.
      Comparing HSV histograms of resized crops is inherently scale-invariant.
    - HSV colour space separates hue/saturation from brightness (Value channel),
      making the comparison more robust to changing court lighting.
    - The bank is populated only from accepted YOLO detections. Once full it is
      locked — no entries are ever replaced or evicted.

    Validation logic:
    - When the bank is empty, any detection is accepted (cold start).
    - Once populated, a new candidate crop must achieve a histogram correlation
      >= min_similarity against at least one stored template to be accepted.
    """

    # Standard crop size used for all histogram comparisons.
    # Small enough to be fast; large enough to capture colour distribution.
    COMPARE_SIZE = (32, 32)

    # HSV histogram bins: 18 hue bins × 16 saturation bins.
    # Value (brightness) is intentionally excluded for lighting robustness.
    H_BINS, S_BINS = 18, 16

    def __init__(self, max_size: int = 8, min_similarity: float = 0.40):
        self.max_size = max_size
        self.min_similarity = min_similarity
        # Each entry: (normalised HSV histogram, (w, h), original BGR crop)
        self._entries: list[tuple[np.ndarray, tuple[int, int], np.ndarray]] = []

    # ------------------------------------------------------------------
    def _crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """Extract and return the ball crop from frame, or None if invalid."""
        x, y, w, h = (int(v) for v in bbox)
        if w <= 0 or h <= 0:
            return None
        crop = frame[y: y + h, x: x + w]
        return crop if crop.size > 0 else None

    def _hist(self, crop: np.ndarray) -> np.ndarray:
        """Compute a normalised 2-D HSV (hue × saturation) histogram."""
        resized = cv2.resize(crop, self.COMPARE_SIZE)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.H_BINS, self.S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    # ------------------------------------------------------------------
    def is_full(self) -> bool:
        return len(self._entries) >= self.max_size

    def add(self, frame: np.ndarray, bbox: tuple) -> bool:
        """
        Add a ball crop to the bank.  Returns False if the crop was invalid or
        the bank is already full (entries are never replaced once saved).
        """
        if self.is_full():
            return False
        crop = self._crop(frame, bbox)
        if crop is None:
            return False
        _, _, w, h = (int(v) for v in bbox)
        self._entries.append((self._hist(crop), (w, h), crop.copy()))
        return True

    def save(self, output_dir: "Path") -> None:
        """
        Write each stored crop to output_dir as template_00.png, template_01.png, …
        Each image is saved at its original resolution.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, (_, (w, h), crop) in enumerate(self._entries):
            path = output_dir / f"template_{i:02d}_{w}x{h}.png"
            cv2.imwrite(str(path), crop)

    def __len__(self) -> int:
        return len(self._entries)

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    # ------------------------------------------------------------------
    def best_similarity(self, frame: np.ndarray, bbox: tuple) -> float:
        """
        Return the highest histogram correlation between the candidate bbox
        and any stored template.  Range: [-1, 1], higher is more similar.
        Returns 1.0 when the bank is empty (no evidence against acceptance).
        """
        if self.is_empty():
            return 1.0
        crop = self._crop(frame, bbox)
        if crop is None:
            return 0.0
        cand_hist = self._hist(crop)
        best = max(
            cv2.compareHist(cand_hist, tmpl_hist, cv2.HISTCMP_CORREL)
            for tmpl_hist, _, _ in self._entries
        )
        return float(best)

    def validate(self, frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
        """
        Check whether a candidate bbox looks like the ball we have been tracking.

        Returns:
            (accepted: bool, similarity_score: float)
        """
        score = self.best_similarity(frame, bbox)
        return score >= self.min_similarity, score

    # ------------------------------------------------------------------
    def scale_range(self) -> tuple[float, float] | None:
        """
        Return the (min, max) diagonal size seen in stored templates.
        Useful for diagnostic output.
        """
        if self.is_empty():
            return None
        diags = [np.hypot(w, h) for _, (w, h), _ in self._entries]
        return float(min(diags)), float(max(diags))


# ---------------------------------------------------------------------------
# CamShift tracker
# ---------------------------------------------------------------------------

class CamShiftTracker:
    """
    Wraps OpenCV CamShift behind the same init/update/getTrackingScore interface
    as TrackerVit and TrackerNano, so it slots into the tracking loop unchanged.

    On each init() call the tracker rebuilds a combined hue histogram from all
    crops currently stored in the TemplateBank.  Because the bank is fixed once
    full, the histogram is stable across the whole video — CamShift is always
    chasing the same target appearance rather than drifting with the scene.

    Tracking score:
      The mean back-projection response inside the result window, normalised to
      [0, 1].  High values mean the pixels in the tracked region closely match
      the stored hue distribution.
    """

    # Pixels below these HSV thresholds are masked out before back-projection
    # so the tracker ignores achromatic (white lines, dark shadows) areas.
    _HSV_LO = np.array([0,  30,  32], dtype=np.uint8)
    _HSV_HI = np.array([180, 255, 255], dtype=np.uint8)

    _TERM_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def __init__(self, bank: "TemplateBank"):
        self._bank = bank
        self._window: tuple = (0, 0, 1, 1)
        self._hist: np.ndarray | None = None
        self._score: float = 0.0

    def _build_hist(self) -> np.ndarray | None:
        """
        Accumulate a 1-D hue histogram (180 bins) across all bank crops.
        Returns None when the bank is empty.
        """
        crops = [crop for _, _, crop in self._bank._entries]
        if not crops:
            return None
        hist = np.zeros((180, 1), dtype=np.float32)
        for crop in crops:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self._HSV_LO, self._HSV_HI)
            hist += cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def init(self, frame: np.ndarray, bbox: tuple) -> bool:
        """Set the initial search window and build the target histogram."""
        x, y, w, h = (int(v) for v in bbox)
        self._window = (x, y, w, h)
        self._hist = self._build_hist()
        self._score = 0.0
        return self._hist is not None

    def update(self, frame: np.ndarray) -> tuple[bool, tuple]:
        """
        Run one CamShift iteration.  Returns (success, (x, y, w, h)).
        The search window is seeded from the previous result so the tracker
        follows the ball across frames.
        """
        if self._hist is None:
            return False, (0, 0, 0, 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._HSV_LO, self._HSV_HI)
        back_proj = cv2.calcBackProject([hsv], [0], self._hist, [0, 180], 1)
        back_proj &= mask

        rotated_rect, new_window = cv2.CamShift(back_proj, self._window, self._TERM_CRIT)
        self._window = new_window

        # Convert the rotated rect to an axis-aligned bbox
        pts = cv2.boxPoints(rotated_rect).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        # Score: mean back-projection response inside the new search window
        wx, wy, ww, wh = new_window
        roi = back_proj[wy: wy + wh, wx: wx + ww]
        self._score = float(roi.mean()) / 255.0 if roi.size > 0 else 0.0

        return True, (x, y, w, h)

    def getTrackingScore(self) -> float:
        return self._score


# ---------------------------------------------------------------------------
# Bounding box sanity checks
# ---------------------------------------------------------------------------

def is_bbox_sane(
    bbox: tuple,
    frame_w: int,
    frame_h: int,
    prev_bbox: tuple | None = None,
    max_area_ratio: float = 0.15,
    max_jump_factor: float = 4.0,
    max_aspect: float = 3.0,
) -> tuple[bool, str]:
    """
    Reject bounding boxes that couldn't physically be a basketball.

    Checks (all must pass):
      1. Area  — bbox must not cover more than max_area_ratio of the frame.
                 Catches full-screen drift that happens when the ball comes
                 very close to the camera and TrackerVit locks onto the
                 background instead.
      2. Aspect ratio — w/h and h/w must both be < max_aspect.
                 A basketball is roughly circular; extreme rectangles are noise.
      3. Size jump — if a previous bbox is known, the new area must not be
                 more than max_jump_factor× larger in a single frame.
                 Catches the moment TrackerVit suddenly expands from a small
                 ball crop to a huge region.

    Returns:
        (ok: bool, reason: str)  — reason is empty string when ok=True.
    """
    x, y, w, h = (int(v) for v in bbox)

    if w <= 0 or h <= 0:
        return False, "zero-size bbox"

    frame_area = frame_w * frame_h
    bbox_area  = w * h
    area_ratio = bbox_area / frame_area

    if area_ratio > max_area_ratio:
        return False, (
            f"bbox too large ({area_ratio*100:.1f}% of frame, "
            f"limit {max_area_ratio*100:.1f}%)"
        )

    aspect = w / h
    if aspect > max_aspect or aspect < 1.0 / max_aspect:
        return False, f"bad aspect ratio ({aspect:.2f})"

    if prev_bbox is not None:
        _, _, pw, ph = (int(v) for v in prev_bbox)
        prev_area = pw * ph
        if prev_area > 0 and bbox_area > prev_area * max_jump_factor:
            return False, (
                f"bbox grew too fast ({bbox_area/prev_area:.1f}× "
                f"vs limit {max_jump_factor:.1f}×)"
            )

    return True, ""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Track a basketball in a video using YOLO + TrackerVit"
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--yolo-model-calibration",
        default=None,
        help="Path to the YOLO model used during the calibration phase. "
             "If not provided, calibration is skipped."
    )
    parser.add_argument(
        "--yolo-imgsz-calibration", type=int, default=640,
        help="Input image size for the calibration YOLO model (default: 640)"
    )
    parser.add_argument(
        "--yolo-model-detection",
        default=None,
        help="Path to the YOLO model used during the detection/tracking phase. "
             "If not provided, YOLO detection is disabled."
    )
    parser.add_argument(
        "--yolo-imgsz-detection", type=int, default=640,
        help="Input image size for the detection YOLO model (default: 640)"
    )
    parser.add_argument(
        "--tracker", choices=["vittrack", "nano", "camshift"], default="vittrack",
        help="Which OpenCV tracker to use: 'vittrack' (default), 'nano', or 'camshift'. "
             "'camshift' requires no model file — it uses the template bank hue histogram."
    )
    parser.add_argument(
        "--vittrack-model",
        default=None,
        help="Path to the TrackerVit ONNX model file. "
             "Required when --tracker vittrack. "
             "If omitted entirely, runs YOLO-only detection on every frame."
    )
    parser.add_argument(
        "--nano-backbone",
        default=None,
        help="Path to the TrackerNano backbone ONNX model file. "
             "Required when --tracker nano."
    )
    parser.add_argument(
        "--nano-neckhead",
        default=None,
        help="Path to the TrackerNano neckhead ONNX model file. "
             "Required when --tracker nano."
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.25,
        help="Tracker confidence below which tracking is considered lost (default: 0.25)"
    )
    parser.add_argument(
        "--yolo-confidence", type=float, default=0.5,
        help="Minimum YOLO detection confidence to accept a detection (default: 0.5)"
    )
    parser.add_argument(
        "--redetect-interval", type=int, default=0,
        help="Force YOLO re-detection every N frames for drift correction. "
             "0 = disabled (default: 0)"
    )
    parser.add_argument(
        "--template-bank-size", type=int, default=8,
        help="Max number of ball crops stored in the template bank (default: 8)"
    )
    parser.add_argument(
        "--template-similarity", type=float, default=0.40,
        help="Minimum HSV histogram correlation for a YOLO detection to pass "
             "template validation (default: 0.40)."
    )
    parser.add_argument(
        "--no-template-validation", action="store_true",
        help="Disable template bank appearance checks entirely. "
             "All sane YOLO detections and tracker results are accepted regardless "
             "of how they compare to stored templates."
    )
    parser.add_argument(
        "--num-templates", type=int, default=5,
        help="(nano only) Number of YOLO detections to collect as frozen templates "
             "before tracking starts (default: 5, minimum: 1)."
    )
    parser.add_argument(
        "--calibration-frames", type=int, default=100,
        help="(nano only) Maximum number of frames to spend collecting templates. "
             "If the target count is not reached, tracking starts with however many "
             "were collected (minimum 1), or exits with an error if zero (default: 100)."
    )
    parser.add_argument(
        "--yolo-periodic", type=int, default=10,
        help="Run YOLO re-detection every N frames as a drift-correction check "
             "(default: 10). Lower values correct faster but cost more inference time."
    )
    parser.add_argument(
        "--search-crops", type=int, default=5,
        help="(nano only) Max search crops per frame. 1=single center only, "
             "5=full motion-predicted multi-crop (default: 5)."
    )
    parser.add_argument(
        "--early-exit-score", type=float, default=0.85,
        help="(nano only) Raw classifier threshold to accept a crop immediately "
             "and skip remaining crops (default: 0.85)."
    )
    parser.add_argument(
        "--motion-history", type=int, default=5,
        help="(nano only) Number of recent positions for velocity estimation "
             "(default: 5)."
    )
    parser.add_argument(
        "--no-motion-mask", action="store_true",
        help="Disable the motion green screen preprocessing. When set, YOLO "
             "receives the raw frame instead of the background-subtracted frame."
    )
    parser.add_argument(
        "--media-pipe",
        default=None,
        help="Path to a MediaPipe ObjectDetector .tflite model file. "
             "When provided, MediaPipe detections are drawn as CYAN bounding "
             "boxes on the output alongside the normal tracking overlay."
    )
    parser.add_argument(
        "--max-bbox-area", type=float, default=0.15,
        help="Reject any bbox whose area exceeds this fraction of the frame "
             "(default: 0.15 = 15%%). Prevents full-screen drift."
    )
    parser.add_argument(
        "--max-bbox-jump", type=float, default=4.0,
        help="Reject a TrackerVit bbox that grew more than this factor vs the "
             "previous frame (default: 4.0). Catches sudden expansion events."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# YOLO detection helper
# ---------------------------------------------------------------------------

def detect_ball_yolo(model, frame, yolo_conf_threshold, input_size=640):
    """
    Run YOLO inference on a single frame and return the best basketball bbox.

    Returns:
        (x, y, w, h) tuple in pixel coords, or None if no ball found.
    """
    results = model.predict(
        source=frame,
        imgsz=input_size,
        conf=yolo_conf_threshold,
        classes=[0],       # basketball is class 0
        verbose=False,
    )

    best_box = None
    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                # xyxy -> x, y, w, h  (OpenCV tracker format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    return best_box


# ---------------------------------------------------------------------------
# Tracker factories
# ---------------------------------------------------------------------------

def create_vittrack_tracker(onnx_model_path: str):
    """
    Create and return a fresh OpenCV TrackerVit instance.
    Requires OpenCV >= 4.9.0 with the tracking contrib module.
    """
    params = cv2.TrackerVit_Params()
    params.net = onnx_model_path
    return cv2.TrackerVit_create(params)


def create_nano_tracker(backbone_path: str, neckhead_path: str,
                        search_crops: int = 5, early_exit_score: float = 0.85,
                        motion_history: int = 5):
    """
    Create and return a fresh OpenCV TrackerNano instance.
    Requires OpenCV >= 4.8.0 with the tracking contrib module.
    TrackerNano uses two separate ONNX files: a backbone and a neckhead.
    """
    params = cv2.TrackerNano_Params()
    params.backbone = backbone_path
    params.neckhead = neckhead_path
    params.searchCrops = search_crops
    params.earlyExitScore = early_exit_score
    params.motionHistory = motion_history
    return cv2.TrackerNano_create(params)



# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_bbox(frame, bbox, color, label):
    """Draw a bounding box with a filled label badge above it."""
    x, y, w, h = (int(v) for v in bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 4),
        (x + text_w + 4, y),
        color, -1,
    )
    cv2.putText(
        frame, label,
        (x + 2, y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
    )


def draw_overlay(frame, frame_num, mode, tracker_conf, bank_size, bank_scale_range):
    """Draw HUD info in the top-left corner."""
    scale_str = (
        f"{bank_scale_range[0]:.0f}-{bank_scale_range[1]:.0f}px"
        if bank_scale_range else "N/A"
    )
    lines = [
        f"Frame:    {frame_num}",
        f"Mode:     {mode}",
        f"TrkConf:  {tracker_conf:.3f}" if tracker_conf is not None else "TrkConf:  N/A",
        f"Bank:     {bank_size} templates  scale={scale_str}",
    ]
    y_start = 24
    for line in lines:
        # White outline + dark fill for readability on any background
        cv2.putText(frame, line, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y_start += 20


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Validate inputs ---------------------------------------------------
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    yolo_calib_path = None
    if args.yolo_model_calibration is not None:
        yolo_calib_path = Path(args.yolo_model_calibration)
        if not yolo_calib_path.exists():
            print(f"[ERROR] YOLO calibration model not found: {yolo_calib_path}", file=sys.stderr)
            sys.exit(1)

    yolo_detect_path = None
    if args.yolo_model_detection is not None:
        yolo_detect_path = Path(args.yolo_model_detection)
        if not yolo_detect_path.exists():
            print(f"[ERROR] YOLO detection model not found: {yolo_detect_path}", file=sys.stderr)
            sys.exit(1)

    # ---- MediaPipe Object Detector (optional) --------------------------------
    mp_detector = None
    mp = None
    if args.media_pipe is not None:
        mp_model_path = Path(args.media_pipe)
        if not mp_model_path.exists():
            print(f"[ERROR] MediaPipe model not found: {mp_model_path}", file=sys.stderr)
            sys.exit(1)
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions as MpBaseOptions
        mp_options = mp_vision.ObjectDetectorOptions(
            base_options=MpBaseOptions(model_asset_path=str(mp_model_path)),
            max_results=10,
            score_threshold=0.3,
        )
        mp_detector = mp_vision.ObjectDetector.create_from_options(mp_options)
        print(f"[INFO] MediaPipe ObjectDetector loaded: {mp_model_path}")

    # Resolve tracker model paths based on chosen tracker type
    vittrack_model_path = None
    nano_backbone_path = None
    nano_neckhead_path = None

    if args.tracker == "vittrack":
        if args.vittrack_model is not None:
            vittrack_model_path = Path(args.vittrack_model)
            if not vittrack_model_path.exists():
                print(f"[ERROR] TrackerVit model not found: {vittrack_model_path}", file=sys.stderr)
                sys.exit(1)
        # else: YOLO-only mode (no tracker)
    elif args.tracker == "nano":
        if args.nano_backbone is None or args.nano_neckhead is None:
            print("[ERROR] --tracker nano requires both --nano-backbone and --nano-neckhead", file=sys.stderr)
            sys.exit(1)
        nano_backbone_path = Path(args.nano_backbone)
        nano_neckhead_path = Path(args.nano_neckhead)
        if not nano_backbone_path.exists():
            print(f"[ERROR] TrackerNano backbone not found: {nano_backbone_path}", file=sys.stderr)
            sys.exit(1)
        if not nano_neckhead_path.exists():
            print(f"[ERROR] TrackerNano neckhead not found: {nano_neckhead_path}", file=sys.stderr)
            sys.exit(1)
    # camshift: no model files required

    # True when any OpenCV tracker is configured
    use_tracker = (
        vittrack_model_path is not None
        or nano_backbone_path is not None
        or args.tracker == "camshift"
    )
    tracker_label = (
        "TrackerVit"  if vittrack_model_path is not None
        else "TrackerNano" if nano_backbone_path is not None
        else "CamShift"    if args.tracker == "camshift"
        else "None"
    )

    # ---- Output path -------------------------------------------------------
    output_path = video_path.with_name(video_path.stem + "_tracked" + video_path.suffix)

    # ---- Load YOLO models ---------------------------------------------------
    yolo_model_calib = None
    yolo_model_detect = None
    if yolo_calib_path is not None:
        print(f"[INFO] Loading YOLO calibration model: {yolo_calib_path}")
        yolo_model_calib = YOLO(str(yolo_calib_path))
    if yolo_detect_path is not None:
        if yolo_detect_path == yolo_calib_path and yolo_model_calib is not None:
            yolo_model_detect = yolo_model_calib
            print(f"[INFO] Using same model for detection")
        else:
            print(f"[INFO] Loading YOLO detection model: {yolo_detect_path}")
            yolo_model_detect = YOLO(str(yolo_detect_path))
    if yolo_model_calib is None and yolo_model_detect is None:
        print(f"[INFO] No YOLO models provided — running without YOLO detection")

    # ---- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {video_path.name}  |  {width}x{height} @ {fps:.1f} fps  |  {total_frames} frames")
    print(f"[INFO] Output: {output_path}")
    if use_tracker:
        print(f"[INFO] Mode: YOLO + {tracker_label}")
        print(f"[INFO] Tracker conf threshold  : {args.confidence_threshold}")
        if args.tracker in ("nano", "vittrack"):
            print(f"[INFO] Calibration templates   : {args.num_templates}")
            print(f"[INFO] Calibration frame limit : {args.calibration_frames}")
    else:
        print(f"[INFO] Mode: YOLO-only (no tracker model provided)")
    print(f"[INFO] YOLO detection conf     : {args.yolo_confidence}")
    print(f"[INFO] Template bank size      : {args.template_bank_size}")
    print(f"[INFO] Template similarity min : {args.template_similarity}")
    print(f"[INFO] Max bbox area           : {args.max_bbox_area*100:.1f}% of frame")
    print(f"[INFO] Max bbox size jump      : {args.max_bbox_jump:.1f}x per frame")
    if args.redetect_interval > 0:
        print(f"[INFO] Forced re-detection every {args.redetect_interval} frames")

    # ---- VideoWriter -------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # ---- Template bank -----------------------------------------------------
    bank = TemplateBank(
        max_size=args.template_bank_size,
        min_similarity=args.template_similarity,
    )

    # ---- Tracker state -----------------------------------------------------
    tracker = None
    tracking_active = False
    bbox = None          # (x, y, w, h)
    tracker_conf = None  # latest tracker score

    # ---- Calibration state (shared by nano and vittrack) --------------------
    # Created once here; never replaced — recovery reuses the same object
    # via initFromEmbedding() so the frozen templates are always intact.
    nano_calibrating = False
    nano_calib_done  = False
    nano_templates_collected = 0
    nano_calib_frame_count   = 0
    nano_template_crops = []   # (crop, w, h, frame_num) saved during calibration
    tracker_templates_dir = output_path.with_name("tracker_templates")
    if args.tracker == "nano":
        tracker = create_nano_tracker(str(nano_backbone_path), str(nano_neckhead_path),
                                     search_crops=args.search_crops,
                                     early_exit_score=args.early_exit_score,
                                     motion_history=args.motion_history)
        nano_calibrating = True
    elif args.tracker == "vittrack" and vittrack_model_path is not None:
        tracker = create_vittrack_tracker(str(vittrack_model_path))
        nano_calibrating = True  # reuse same calibration flow

    # ---- Calibration size reference for diagnostics -------------------------
    calib_bbox_sizes = []   # list of (w, h) from calibration detections

    # ---- Counters ----------------------------------------------------------
    frames_vittrack = 0
    frames_yolo = 0
    frames_no_ball = 0
    tracking_lost_count = 0
    detections_rejected_by_bank = 0
    bboxes_rejected_by_sanity = 0

    # Run YOLO on frame 1 and then as a sanity check every 10 frames
    YOLO_PERIODIC = args.yolo_periodic

    # ---- Motion green screen state ------------------------------------------
    previous_frame_gray = None
    motion_kernel = np.ones((5, 5), np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    MOTION_MIN_AREA = 100
    MOTION_MAX_AREA = 3000
    motion_mask_dir = output_path.with_name("motion-mask")
    if not args.no_motion_mask:
        motion_mask_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Processing — press 'q' in the display window to quit early.\n")

    frame_num = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ---- Motion green screen: prepare grayscale for this frame -----------
        if not args.no_motion_mask:
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_frame_gray is None:
                previous_frame_gray = current_frame_gray
                # Write the raw first frame to output and skip processing
                writer.write(frame)
                cv2.imshow("Basketball Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Quit requested by user.")
                    break
                continue

        mode = "YOLO"
        current_color = (0, 0, 255)   # RED  = YOLO active
        bbox_to_draw = None

        # ----------------------------------------------------------------
        # Nano calibration phase — runs instead of Phase 1/2 until done
        # ----------------------------------------------------------------
        if nano_calibrating:
            nano_calib_frame_count += 1
            detected_bbox = (
                detect_ball_yolo(yolo_model_calib, frame, args.yolo_confidence, args.yolo_imgsz_calibration)
                if yolo_model_calib is not None else None
            )

            if detected_bbox is not None:
                sane, sane_reason = is_bbox_sane(
                    detected_bbox, width, height,
                    prev_bbox=bbox,
                    max_area_ratio=args.max_bbox_area,
                    max_jump_factor=args.max_bbox_jump,
                )
                if sane:
                    tracker.addTemplate(frame, detected_bbox)
                    bank.add(frame, detected_bbox)
                    nano_templates_collected += 1
                    bbox = detected_bbox
                    bbox_to_draw = bbox
                    calib_bbox_sizes.append((int(detected_bbox[2]), int(detected_bbox[3])))
                    # Save calibration crop
                    bx, by, bw, bh = (int(v) for v in detected_bbox)
                    crop = frame[by:by+bh, bx:bx+bw]
                    if crop.size > 0:
                        nano_template_crops.append((crop.copy(), bw, bh, frame_num))
                else:
                    bboxes_rejected_by_sanity += 1

            if bbox_to_draw is not None:
                mode = f"CALIBRATING ({nano_templates_collected}/{args.num_templates})"
                current_color = (255, 128, 0)   # BLUE

            # Finalize when target reached or calibration window exhausted
            calib_done = (
                nano_templates_collected >= args.num_templates
                or nano_calib_frame_count >= args.calibration_frames
            )
            if calib_done:
                if nano_templates_collected == 0:
                    print("[ERROR] Calibration failed: no ball detected in the "
                          f"first {args.calibration_frames} frames.", file=sys.stderr)
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    sys.exit(1)
                tracker.finalizeTemplates()
                tracker.init(frame, bbox)
                tracking_active = True
                nano_calibrating = False
                nano_calib_done  = True
                # Save calibration crops to disk
                tracker_templates_dir.mkdir(parents=True, exist_ok=True)
                for i, (crop, cw, ch, fnum) in enumerate(nano_template_crops):
                    path = tracker_templates_dir / f"template_{i:02d}_f{fnum}_{cw}x{ch}.png"
                    cv2.imwrite(str(path), crop)
                print(
                    f"[INFO] Calibration complete: {nano_templates_collected} template(s) "
                    f"collected in {nano_calib_frame_count} frame(s) — tracking started."
                )
                print(f"[INFO] {len(nano_template_crops)} template crop(s) saved to: {tracker_templates_dir}")

            # Skip normal Phase 1 / Phase 2 during calibration
            output_frame = frame.copy()
            if bbox_to_draw is not None:
                draw_bbox(output_frame, bbox_to_draw, current_color, mode)
            draw_overlay(output_frame, frame_num, mode, tracker_conf,
                         len(bank), bank.scale_range())
            writer.write(output_frame)
            cv2.imshow("Basketball Tracker", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Quit requested by user.")
                break
            continue

        # ----------------------------------------------------------------
        # Should YOLO run this frame?
        # ----------------------------------------------------------------
        run_yolo = (
            not tracking_active
            or frame_num % YOLO_PERIODIC == 0
            or (args.redetect_interval > 0 and frame_num % args.redetect_interval == 0)
        )

        # ----------------------------------------------------------------
        # Phase 1: OpenCV tracker update (skipped in YOLO-only mode)
        # ----------------------------------------------------------------
        if not use_tracker:
            run_yolo = True  # always detect with YOLO when there is no tracker

        if tracking_active and tracker is not None and use_tracker:
            success, trk_bbox = tracker.update(frame)
            tracker_conf = tracker.getTrackingScore()

            # --- Diagnostic: print tracking score, bbox position and size ---
            tx, ty, tw, th = (int(v) for v in trk_bbox)
            print(
                f"  [DIAG] Frame {frame_num}: tracker score={tracker_conf:.4f} "
                f"pos=({tx},{ty}) size=({tw}x{th})"
            )
            # Compare tracked bbox size against calibration template sizes
            if calib_bbox_sizes:
                calib_diags = [np.hypot(cw, ch) for cw, ch in calib_bbox_sizes]
                avg_calib_diag = float(np.mean(calib_diags))
                trk_diag = float(np.hypot(tw, th))
                size_ratio = trk_diag / avg_calib_diag if avg_calib_diag > 0 else 0
                if size_ratio > 2.0 or size_ratio < 0.3:
                    print(
                        f"  [DIAG][WARNING] Frame {frame_num}: tracked bbox size "
                        f"({tw}x{th}, diag={trk_diag:.0f}) deviates from calibration "
                        f"avg diag={avg_calib_diag:.0f} (ratio={size_ratio:.2f})"
                    )
            # Every 30 frames: compare YOLO detection vs tracker position
            if frame_num % 30 == 0 and yolo_model_detect is not None:
                yolo_bbox_check = detect_ball_yolo(yolo_model_detect, frame, args.yolo_confidence, args.yolo_imgsz_detection)
                if yolo_bbox_check is not None:
                    yx, yy, yw, yh = yolo_bbox_check
                    yolo_cx, yolo_cy = yx + yw / 2, yy + yh / 2
                    trk_cx, trk_cy = tx + tw / 2, ty + th / 2
                    drift = np.hypot(yolo_cx - trk_cx, yolo_cy - trk_cy)
                    print(
                        f"  [DIAG][DRIFT] Frame {frame_num}: "
                        f"YOLO=({yx},{yy},{yw}x{yh}) center=({yolo_cx:.0f},{yolo_cy:.0f}) | "
                        f"Tracker=({tx},{ty},{tw}x{th}) center=({trk_cx:.0f},{trk_cy:.0f}) | "
                        f"drift={drift:.1f}px"
                    )
                else:
                    print(
                        f"  [DIAG][DRIFT] Frame {frame_num}: YOLO found nothing, "
                        f"tracker at ({tx},{ty},{tw}x{th})"
                    )

            # Sanity-check the returned bbox BEFORE accepting it.
            # Some trackers can drift to cover the full screen (e.g. when the
            # ball comes very close and then disappears) while still reporting
            # high confidence — this guard catches that case.
            sane, sane_reason = is_bbox_sane(
                trk_bbox, width, height,
                prev_bbox=bbox,
                max_area_ratio=args.max_bbox_area,
                max_jump_factor=args.max_bbox_jump,
            )

            # Gate: appearance — tracked region must still look like the ball.
            # This catches tracker drift onto non-ball content (e.g. when the
            # ball disappears and the tracker latches onto a jersey or the floor).
            if args.no_template_validation:
                trk_accepted, trk_similarity = True, 1.0
            elif sane:
                trk_accepted, trk_similarity = bank.validate(frame, trk_bbox)
            else:
                trk_accepted, trk_similarity = False, 0.0

            if success and tracker_conf >= args.confidence_threshold and sane and trk_accepted:
                bbox = trk_bbox
                mode = tracker_label
                current_color = (0, 255, 0)   # GREEN = tracker active
                bbox_to_draw = bbox
                frames_vittrack += 1
                # For calibrated trackers (nano/vittrack), always allow periodic
                # YOLO re-detection as a drift-correction mechanism.
                if not nano_calib_done:
                    run_yolo = False   # tracking is healthy — override periodic check
            else:
                # Tracking confidence dropped, bbox implausible, or drifted off-ball
                reason_parts = []
                if not (success and tracker_conf >= args.confidence_threshold):
                    reason_parts.append(f"conf={tracker_conf:.3f}")
                if not sane:
                    reason_parts.append(f"sanity={sane_reason}")
                    bboxes_rejected_by_sanity += 1
                if sane and not trk_accepted:
                    reason_parts.append(f"appearance={trk_similarity:.3f}")

                tracking_active = False
                # Keep nano tracker alive — recovery uses initFromEmbedding()
                # on the same object so frozen templates remain intact.
                if not nano_calib_done:
                    tracker = None
                tracking_lost_count += 1
                run_yolo = True
                print(
                    f"  [WARN] Frame {frame_num}: {tracker_label} lost "
                    f"({', '.join(reason_parts)}) — falling back to YOLO"
                )

        # ----------------------------------------------------------------
        # Phase 2: YOLO detection (raw frame first, motion mask fallback)
        # ----------------------------------------------------------------
        if run_yolo:
            tracker_conf = None

            # --- Try YOLO on the raw frame first ---
            detected_bbox = (
                detect_ball_yolo(yolo_model_detect, frame, args.yolo_confidence, args.yolo_imgsz_detection)
                if yolo_model_detect is not None else None
            )
            used_motion_mask = False

            if detected_bbox is not None:
                sane, sane_reason = is_bbox_sane(
                    detected_bbox, width, height,
                    prev_bbox=bbox,
                    max_area_ratio=args.max_bbox_area,
                    max_jump_factor=args.max_bbox_jump,
                )
                if not sane:
                    bboxes_rejected_by_sanity += 1
                    print(
                        f"  [INFO] Frame {frame_num}: YOLO bbox rejected by sanity "
                        f"check ({sane_reason})"
                    )
                    detected_bbox = None

            # --- Fallback: motion mask if raw frame found nothing ---
            if detected_bbox is None and yolo_model_detect is not None and not args.no_motion_mask and previous_frame_gray is not None:
                frame_diff = cv2.absdiff(previous_frame_gray, current_frame_gray)
                _, motion_bin = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                motion_bin = cv2.dilate(motion_bin, motion_kernel, iterations=3)

                closed_mask = cv2.morphologyEx(motion_bin, cv2.MORPH_CLOSE, close_kernel)
                contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cleaned_mask = np.zeros_like(motion_bin)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < MOTION_MIN_AREA:
                        continue
                    elif area <= MOTION_MAX_AREA:
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        cv2.circle(cleaned_mask, (int(cx), int(cy)), int(radius), 255, -1)
                    else:
                        cv2.drawContours(cleaned_mask, [cnt], -1, 255, -1)

                yolo_frame = cv2.bitwise_and(frame, frame, mask=cleaned_mask)
                cv2.imwrite(str(motion_mask_dir / f"frame_{frame_num:06d}.png"), yolo_frame)

                detected_bbox = detect_ball_yolo(yolo_model_detect, yolo_frame, args.yolo_confidence, args.yolo_imgsz_detection)
                used_motion_mask = True

                if detected_bbox is not None:
                    sane, sane_reason = is_bbox_sane(
                        detected_bbox, width, height,
                        prev_bbox=bbox,
                        max_area_ratio=args.max_bbox_area,
                        max_jump_factor=args.max_bbox_jump,
                    )
                    if not sane:
                        bboxes_rejected_by_sanity += 1
                        print(
                            f"  [INFO] Frame {frame_num}: YOLO (motion mask) bbox rejected "
                            f"by sanity check ({sane_reason})"
                        )
                        detected_bbox = None

            if detected_bbox is not None:
                # Gate 2: appearance — must look like the ball we've been tracking.
                # Skipped in YOLO-only mode: with no tracker there is no
                # inter-frame continuity to protect, so every sane YOLO
                # detection is accepted directly.
                if use_tracker and not args.no_template_validation:
                    accepted, similarity = bank.validate(frame, detected_bbox)
                else:
                    accepted, similarity = True, 1.0

                if accepted:
                    bbox = detected_bbox
                    bbox_to_draw = bbox
                    mode = "YOLO+MASK" if used_motion_mask else "YOLO"
                    current_color = (0, 0, 255)   # RED
                    frames_yolo += 1

                    # Feed accepted detection into the template bank
                    bank.add(frame, bbox)

                    # (Re-)initialise the OpenCV tracker from this detection
                    if use_tracker:
                        if nano_calib_done:
                            # Re-initialize tracker from the current YOLO detection.
                            # This re-encodes the template from the current frame,
                            # keeping the template fresh as the ball changes appearance.
                            tracker.init(frame, bbox)
                            tracking_active = True
                        else:
                            tracker = CamShiftTracker(bank)
                            if tracker.init(frame, bbox) is not False:
                                tracking_active = True
                            else:
                                tracker = None  # bank empty — CamShift can't start yet
                else:
                    # YOLO found something but it doesn't look like our ball
                    detections_rejected_by_bank += 1
                    frames_no_ball += 1
                    bbox_to_draw = None   # don't draw rejected boxes by default
                    tracking_active = False
                    print(
                        f"  [INFO] Frame {frame_num}: YOLO detection rejected by template bank "
                        f"(similarity={similarity:.3f} < {args.template_similarity})"
                    )
            else:
                frames_no_ball += 1
                bbox_to_draw = None
                tracking_active = False

        # ----------------------------------------------------------------
        # Draw results
        # ----------------------------------------------------------------
        output_frame = frame.copy()

        # MediaPipe Object Detection overlay (CYAN boxes)
        if mp_detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_result = mp_detector.detect(mp_image)
            for det in mp_result.detections:
                bb = det.bounding_box
                mp_label = det.categories[0].category_name if det.categories else "?"
                mp_score = det.categories[0].score if det.categories else 0.0
                draw_bbox(output_frame,
                          (bb.origin_x, bb.origin_y, bb.width, bb.height),
                          (255, 255, 0),  # CYAN
                          f"MP:{mp_label} {mp_score:.2f}")

        if bbox_to_draw is not None:
            draw_bbox(output_frame, bbox_to_draw, current_color, mode)

        draw_overlay(
            output_frame, frame_num, mode, tracker_conf,
            len(bank), bank.scale_range(),
        )

        # ----------------------------------------------------------------
        # Write & display
        # ----------------------------------------------------------------
        writer.write(output_frame)
        cv2.imshow("Basketball Tracker", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit requested by user.")
            break

        # Update motion green screen state for next iteration
        if not args.no_motion_mask:
            previous_frame_gray = current_frame_gray.copy()

        if frame_num % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
            print(
                f"  Frame {frame_num}/{total_frames} ({pct:.1f}%)  "
                f"speed={fps_proc:.1f} fps  bank={len(bank)}/{args.template_bank_size}"
            )

    # ---- Cleanup -----------------------------------------------------------
    elapsed = time.time() - start_time
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ---- Summary -----------------------------------------------------------
    total_processed = frames_vittrack + frames_yolo + frames_no_ball
    print("\n" + "=" * 58)
    print("  TRACKING SUMMARY")
    print("=" * 58)
    print(f"  Total frames processed      : {total_processed}")
    tracker_label_summary = tracker_label if use_tracker else "OpenCV tracker"
    print(f"  Tracked by {tracker_label_summary:<17} : {frames_vittrack}")
    print(f"  Detected by YOLO            : {frames_yolo}")
    print(f"  No ball found               : {frames_no_ball}")
    print(f"  Tracking lost events        : {tracking_lost_count}")
    print(f"  YOLO detections rejected    : {detections_rejected_by_bank}  (template bank)")
    print(f"  Bboxes rejected by sanity   : {bboxes_rejected_by_sanity}  (size/aspect/jump)")
    if nano_calib_done:
        print(f"  Calibration templates       : {nano_templates_collected}")
        print(f"  Calibration frames used     : {nano_calib_frame_count}")
    print(f"  Total elapsed time          : {elapsed:.1f}s")
    if elapsed > 0:
        print(f"  Average speed               : {total_processed / elapsed:.1f} fps")
    print("=" * 58)
    print(f"\n[INFO] Output saved to: {output_path}")

    # ---- Save template bank crops ------------------------------------------
    if len(bank) > 0:
        templates_dir = output_path.with_name(output_path.stem + "_templates")
        bank.save(templates_dir)
        print(f"[INFO] {len(bank)} template(s) saved to: {templates_dir}")


if __name__ == "__main__":
    main()
