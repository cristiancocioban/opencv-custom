"""
ball_tracker.py - Basketball tracking using the C++ BallTracker class
(YOLO detection + TrackerNano, all running inside OpenCV)

This script is a thin wrapper: it reads video frames, passes them to
cv2.BallTracker.processFrame(), and draws the results.

Color coding:
  GREEN bounding box  = TrackerNano is tracking
  RED bounding box    = YOLO detected the ball
  ORANGE bounding box = Calibration phase
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_bbox(frame, bbox, color, label):
    """Draw a bounding box with a filled label badge above it."""
    x, y, w, h = bbox
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


def draw_overlay(frame, frame_num, mode_str, confidence):
    """Draw HUD info in the top-left corner."""
    lines = [
        f"Frame:    {frame_num}",
        f"Mode:     {mode_str}",
        f"Conf:     {confidence:.3f}" if confidence >= 0 else "Conf:     N/A",
    ]
    y_start = 24
    for line in lines:
        cv2.putText(frame, line, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y_start += 20


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

MODE_COLORS = {
    cv2.BALL_TRACKER_MODE_CALIBRATING: (255, 128, 0),  # ORANGE
    cv2.BALL_TRACKER_MODE_TRACKER:     (0, 255, 0),    # GREEN
    cv2.BALL_TRACKER_MODE_YOLO:        (0, 0, 255),    # RED
    cv2.BALL_TRACKER_MODE_LOST:        (128, 128, 128),# GRAY
}

MODE_LABELS = {
    cv2.BALL_TRACKER_MODE_CALIBRATING: "CALIBRATING",
    cv2.BALL_TRACKER_MODE_TRACKER:     "TrackerNano",
    cv2.BALL_TRACKER_MODE_YOLO:        "YOLO",
    cv2.BALL_TRACKER_MODE_LOST:        "LOST",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Track a basketball in a video using YOLO + TrackerNano (C++ BallTracker)"
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")

    # YOLO models
    parser.add_argument("--yolo-model-calibration", default=None,
                        help="Path to the YOLO model for calibration phase")
    parser.add_argument("--yolo-imgsz-calibration", type=int, default=640,
                        help="Input image size for calibration YOLO model (default: 640)")
    parser.add_argument("--yolo-model-detection", default=None,
                        help="Path to the YOLO model for detection phase")
    parser.add_argument("--yolo-imgsz-detection", type=int, default=640,
                        help="Input image size for detection YOLO model (default: 640)")
    parser.add_argument("--yolo-confidence", type=float, default=0.5,
                        help="Minimum YOLO detection confidence (default: 0.5)")

    # TrackerNano
    parser.add_argument("--nano-backbone", default=None,
                        help="Path to TrackerNano backbone ONNX model")
    parser.add_argument("--nano-neckhead", default=None,
                        help="Path to TrackerNano neckhead ONNX model")
    parser.add_argument("--search-crops", type=int, default=5,
                        help="Max search crops per frame (default: 5)")
    parser.add_argument("--early-exit-score", type=float, default=0.85,
                        help="Crop acceptance threshold (default: 0.85)")
    parser.add_argument("--motion-history", type=int, default=5,
                        help="Velocity estimation history length (default: 5)")
    parser.add_argument("--confidence-threshold", type=float, default=0.25,
                        help="Tracker confidence threshold (default: 0.25)")

    # Calibration
    parser.add_argument("--num-templates", type=int, default=5,
                        help="Number of calibration templates to collect (default: 5)")
    parser.add_argument("--calibration-frames", type=int, default=100,
                        help="Max frames for calibration (default: 100)")

    # Re-detection
    parser.add_argument("--yolo-periodic", type=int, default=10,
                        help="YOLO re-detection every N frames (default: 10)")
    parser.add_argument("--redetect-interval", type=int, default=0,
                        help="Force YOLO re-detection interval, 0=disabled (default: 0)")

    # Sanity
    parser.add_argument("--max-bbox-area", type=float, default=0.15,
                        help="Max bbox area as fraction of frame (default: 0.15)")
    parser.add_argument("--max-bbox-jump", type=float, default=4.0,
                        help="Max bbox area growth per frame (default: 4.0)")

    # Template bank
    parser.add_argument("--template-bank-size", type=int, default=8,
                        help="Max stored templates (default: 8)")
    parser.add_argument("--template-similarity", type=float, default=0.40,
                        help="Min HSV histogram correlation (default: 0.40)")
    parser.add_argument("--no-template-validation", action="store_true",
                        help="Disable template bank appearance checks")

    # DNN backend
    parser.add_argument("--backend", type=str, default="default",
                        choices=["default", "opencv", "inference_engine"],
                        help="DNN backend (default: default)")
    parser.add_argument("--target", type=str, default="cpu",
                        choices=["cpu", "opencl", "opencl_fp16"],
                        help="DNN target device (default: cpu)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Validate video path -----------------------------------------------
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # ---- Build BallTrackerParams -------------------------------------------
    params = cv2.BallTrackerParams()

    if args.yolo_model_calibration:
        params.yoloModelCalibration = args.yolo_model_calibration
    if args.yolo_model_detection:
        params.yoloModelDetection = args.yolo_model_detection
    params.yoloImgszCalibration = args.yolo_imgsz_calibration
    params.yoloImgszDetection = args.yolo_imgsz_detection
    params.yoloConfidence = args.yolo_confidence

    if args.nano_backbone:
        params.nanoBackbone = args.nano_backbone
    if args.nano_neckhead:
        params.nanoNeckhead = args.nano_neckhead
    params.searchCrops = args.search_crops
    params.earlyExitScore = args.early_exit_score
    params.motionHistory = args.motion_history
    params.confidenceThreshold = args.confidence_threshold

    params.numTemplates = args.num_templates
    params.calibrationFrames = args.calibration_frames
    params.yoloPeriodic = args.yolo_periodic
    params.redetectInterval = args.redetect_interval

    params.maxBboxArea = args.max_bbox_area
    params.maxBboxJump = args.max_bbox_jump

    params.templateBankSize = args.template_bank_size
    params.templateSimilarity = args.template_similarity
    params.noTemplateValidation = args.no_template_validation

    backend_map = {
        "default": cv2.dnn.DNN_BACKEND_DEFAULT,
        "opencv": cv2.dnn.DNN_BACKEND_OPENCV,
        "inference_engine": cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
    }
    target_map = {
        "cpu": cv2.dnn.DNN_TARGET_CPU,
        "opencl": cv2.dnn.DNN_TARGET_OPENCL,
        "opencl_fp16": cv2.dnn.DNN_TARGET_OPENCL_FP16,
    }
    params.backend = backend_map[args.backend]
    params.target = target_map[args.target]

    # ---- Create BallTracker ------------------------------------------------
    print("[INFO] Creating BallTracker...")
    tracker = cv2.BallTracker.create(params)

    # ---- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video_path.with_name(video_path.stem + "_tracked" + video_path.suffix)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"[INFO] Video: {video_path.name}  |  {width}x{height} @ {fps:.1f} fps  |  {total_frames} frames")
    print(f"[INFO] Output: {output_path}")
    print(f"\n[INFO] Processing — press 'q' in the display window to quit early.\n")

    # ---- Counters ----------------------------------------------------------
    frames_tracker = 0
    frames_yolo = 0
    frames_lost = 0
    start_time = time.time()

    # ---- Main loop ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.processFrame(frame)

        # Count
        if result.mode == cv2.BALL_TRACKER_MODE_TRACKER:
            frames_tracker += 1
        elif result.mode == cv2.BALL_TRACKER_MODE_YOLO:
            frames_yolo += 1
        elif result.mode == cv2.BALL_TRACKER_MODE_LOST:
            frames_lost += 1

        # Draw
        output_frame = frame.copy()
        mode_str = MODE_LABELS.get(result.mode, "UNKNOWN")
        color = MODE_COLORS.get(result.mode, (255, 255, 255))

        if result.found:
            draw_bbox(output_frame, result.bbox, color, mode_str)

        draw_overlay(output_frame, result.frameNumber, mode_str, result.confidence)

        writer.write(output_frame)
        cv2.imshow("Basketball Tracker", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit requested by user.")
            break

        if result.frameNumber % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = result.frameNumber / elapsed if elapsed > 0 else 0
            pct = (result.frameNumber / total_frames * 100) if total_frames > 0 else 0
            print(f"  Frame {result.frameNumber}/{total_frames} ({pct:.1f}%)  speed={fps_proc:.1f} fps")

    # ---- Cleanup -----------------------------------------------------------
    elapsed = time.time() - start_time
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_processed = tracker.getFrameCount()
    print("\n" + "=" * 58)
    print("  TRACKING SUMMARY")
    print("=" * 58)
    print(f"  Total frames processed      : {total_processed}")
    print(f"  Tracked by TrackerNano      : {frames_tracker}")
    print(f"  Detected by YOLO            : {frames_yolo}")
    print(f"  No ball found               : {frames_lost}")
    print(f"  Total elapsed time          : {elapsed:.1f}s")
    if elapsed > 0:
        print(f"  Average speed               : {total_processed / elapsed:.1f} fps")
    print("=" * 58)
    print(f"\n[INFO] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
