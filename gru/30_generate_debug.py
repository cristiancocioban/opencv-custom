import argparse
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

WINDOW_SIZE = 15
ROI_SIZE = 600


class HoopsWorldModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 3), nn.Sigmoid(),
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_summary = gru_out[:, -1, :]
        return self.classifier_head(final_summary), self.predictor_head(final_summary)


def safe_lm(landmarks, idx):
    if landmarks is None:
        return None
    lm = landmarks[idx]
    return float(lm.x), float(lm.y), float(lm.visibility)


def avg_xy(a, b):
    if a is None or b is None:
        return None
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def detect_best_ball(yolo, image, x_offset=0, y_offset=0, conf_threshold=0.25):
    """Run YOLO on `image` and return ((x1,y1,x2,y2), conf) of the best class-0
    detection in ORIGINAL frame coords, or None if nothing is above threshold."""
    results = yolo(image, classes=[0], verbose=False)
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    confs = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()
    best = int(np.argmax(confs))
    if confs[best] < conf_threshold:
        return None
    x1, y1, x2, y2 = xyxy[best]
    return (
        (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset),
        float(confs[best]),
    )


def draw_ball_box(frame, box, conf, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv2.rectangle(frame, p1, p2, color, 3)
    label = f"Ball {conf:.2f}"
    cv2.putText(frame, label, (p1[0], max(0, p1[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def build_features(lms, idxs, ball_norm, prev_features):
    """Build the 26-feature vector matching the world model's training contract.

    Index 25 is Ball_Detected (1.0 if `ball_norm` is not None, else 0.0).
    """
    features = [0.0] * 26
    hip_center = None
    if not lms:
        return features, hip_center

    lhip = safe_lm(lms, idxs['LHIP'])
    rhip = safe_lm(lms, idxs['RHIP'])
    hip_center = avg_xy(lhip, rhip)
    if hip_center is None:
        return features, hip_center

    hcx, hcy = hip_center

    def rel(lm):
        return (lm[0] - hcx, lm[1] - hcy, lm[2]) if lm else (0.0, 0.0, 0.0)

    if ball_norm:
        features[0], features[1] = ball_norm[0] - hcx, ball_norm[1] - hcy

    features[2:5]   = rel(safe_lm(lms, idxs['LELB']))
    features[5:8]   = rel(safe_lm(lms, idxs['RELB']))
    l_wrist = rel(safe_lm(lms, idxs['LWR']))
    r_wrist = rel(safe_lm(lms, idxs['RWR']))
    features[8:11]  = l_wrist
    features[11:14] = r_wrist
    features[14:17] = rel(safe_lm(lms, idxs['LANK']))
    features[17:20] = rel(safe_lm(lms, idxs['RANK']))

    lsh = safe_lm(lms, idxs['LSH'])
    rsh = safe_lm(lms, idxs['RSH'])
    sh_center = avg_xy(lsh, rsh)
    if sh_center:
        features[20] = abs(sh_center[1] - hcy)

    features[21] = np.sqrt((features[0] - l_wrist[0]) ** 2 + (features[1] - l_wrist[1]) ** 2)
    features[22] = np.sqrt((features[0] - r_wrist[0]) ** 2 + (features[1] - r_wrist[1]) ** 2)

    if prev_features is not None:
        features[23] = features[1] - prev_features[1]
        features[24] = features[0] - prev_features[0]

    features[25] = 1.0 if ball_norm else 0.0

    return features, hip_center


def create_ball_tracker(weights_path, nano_backbone, nano_neckhead):
    """Create a cv2.BallTracker configured to use `weights_path` for both
    calibration and detection, plus the TrackerNano backbone/neckhead."""
    params = cv2.BallTrackerParams()
    params.yoloModelCalibration = weights_path
    params.yoloModelDetection = weights_path
    params.yoloImgszCalibration = 640
    params.yoloImgszDetection = 640
    params.yoloConfidence = 0.25
    params.nanoBackbone = nano_backbone
    params.nanoNeckhead = nano_neckhead
    params.searchCrops = 1
    params.numTemplates = 1
    params.calibrationFrames = 100
    params.yoloPeriodic = 3
    params.maxBboxJump = 100.0
    params.driftThreshold = 0.50
    return cv2.BallTracker.create(params)


def create_debug_video(input_video_path, output_video_path, weights_path,
                      world_model_path, nano_backbone=None, nano_neckhead=None):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    use_tracker = bool(nano_backbone and nano_neckhead)
    yolo = None
    tracker = None
    if use_tracker:
        print(f"Creating BallTracker (YOLO weights: {weights_path}, "
              f"backbone: {nano_backbone}, neckhead: {nano_neckhead})")
        tracker = create_ball_tracker(weights_path, nano_backbone, nano_neckhead)
    else:
        print(f"Loading YOLO weights from {weights_path}")
        yolo = YOLO(weights_path)

    use_world_model = bool(world_model_path)
    world_model = None
    device = None
    if use_world_model:
        print(f"Loading world model from {world_model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_model = HoopsWorldModel(input_size=26)
        world_model.load_state_dict(
            torch.load(world_model_path, map_location=device, weights_only=True)
        )
        world_model.to(device).eval()
    else:
        print("No --world-model provided; running full-frame YOLO detection only.")

    LM = mp_pose.PoseLandmark
    idxs = {
        'LSH': LM.LEFT_SHOULDER.value, 'RSH': LM.RIGHT_SHOULDER.value,
        'LHIP': LM.LEFT_HIP.value,     'RHIP': LM.RIGHT_HIP.value,
        'LELB': LM.LEFT_ELBOW.value,   'RELB': LM.RIGHT_ELBOW.value,
        'LWR':  LM.LEFT_WRIST.value,   'RWR':  LM.RIGHT_WRIST.value,
        'LANK': LM.LEFT_ANKLE.value,   'RANK': LM.RIGHT_ANKLE.value,
    }

    print(f"Processing video: {width}x{height} at {fps} FPS...")

    feature_window = deque(maxlen=WINDOW_SIZE)
    pending_prediction_px = None  # (cx, cy) predicted center for the CURRENT frame

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ---- Ball detection ----------------------------------------------
            ball_box = None
            ball_conf = None
            roi_rect = None
            detected_in_roi = False
            tracker_mode = None

            if use_tracker:
                # BallTracker handles tracking internally (YOLO + TrackerNano).
                # The world-model ROI logic is bypassed in this mode.
                result = tracker.processFrame(frame)
                if result.found:
                    bx, by, bw, bh = result.bbox
                    ball_box = (bx, by, bx + bw, by + bh)
                    ball_conf = result.confidence
                    tracker_mode = result.mode
            else:
                # YOLO + ROI guided by the world-model prediction
                if pending_prediction_px is not None:
                    px, py = pending_prediction_px
                    half = ROI_SIZE // 2
                    x1 = max(0, int(px - half))
                    y1 = max(0, int(py - half))
                    x2 = min(width,  int(px + half))
                    y2 = min(height, int(py + half))
                    if x2 > x1 and y2 > y1:
                        roi_rect = (x1, y1, x2, y2)
                        roi = frame[y1:y2, x1:x2]
                        res = detect_best_ball(yolo, roi, x_offset=x1, y_offset=y1)
                        if res is not None:
                            ball_box, ball_conf = res
                            detected_in_roi = True

                # Fallback: full-frame detection if the ROI search came up empty
                if ball_box is None:
                    res = detect_best_ball(yolo, frame)
                    if res is not None:
                        ball_box, ball_conf = res

                # Visualize the ROI used for the prediction-guided search
                if roi_rect is not None:
                    rx1, ry1, rx2, ry2 = roi_rect
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
                    cv2.putText(frame, "ROI", (rx1, max(0, ry1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            ball_norm = None
            if ball_box is not None:
                bx1, by1, bx2, by2 = ball_box
                ball_norm = ((bx1 + bx2) * 0.5 / width, (by1 + by2) * 0.5 / height)
                if use_tracker:
                    # GREEN=tracker, RED=YOLO, ORANGE=calibrating, GRAY=lost
                    if tracker_mode == cv2.BALL_TRACKER_MODE_TRACKER:
                        color = (0, 255, 0)
                    elif tracker_mode == cv2.BALL_TRACKER_MODE_YOLO:
                        color = (0, 0, 255)
                    elif tracker_mode == cv2.BALL_TRACKER_MODE_CALIBRATING:
                        color = (0, 165, 255)
                    else:
                        color = (128, 128, 128)
                else:
                    color = (0, 255, 0) if detected_in_roi else (0, 165, 255)
                draw_ball_box(frame, ball_box, ball_conf, color=color)

            # ---- MediaPipe pose ----
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            lms = results.pose_landmarks.landmark if results.pose_landmarks else None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                )

            # ---- World model: build features, slide window, predict next ball ----
            pending_prediction_px = None
            if use_world_model:
                prev_features = feature_window[-1] if feature_window else None
                features, hip_center = build_features(lms, idxs, ball_norm, prev_features)
                feature_window.append(features)

                if len(feature_window) == WINDOW_SIZE and hip_center is not None:
                    x_seq = torch.FloatTensor(np.array(feature_window)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, pred_coords = world_model(x_seq)
                    next_x_rel, next_y_rel = pred_coords[0].cpu().numpy()
                    pending_prediction_px = (
                        (next_x_rel + hip_center[0]) * width,
                        (next_y_rel + hip_center[1]) * height,
                    )

            out.write(frame)

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Debug video successfully saved to {output_video_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       default="../raw_dribbling_videos/10_minute_home_dribbling_pound_crossover.mp4")
    ap.add_argument("--output",      default="../raw_dribbling_videos/debug_videos/10_minute_home_dribbling_pound_crossover_debug.mp4")
    ap.add_argument("--weights",     default="../models/ball_detection_v26n_640_07_04_raw.onnx")
    ap.add_argument("--world-model", default=None,
                    help="Optional GRU world-model weights (.pth). If omitted, "
                         "runs full-frame YOLO detection without ROI guidance.")
    ap.add_argument("--tracker", action="store_true",
                    help="Use the custom cv2.BallTracker (YOLO + TrackerNano) "
                         "instead of stand-alone YOLO. Requires --nano-backbone "
                         "and --nano-neckhead.")
    ap.add_argument("--nano-backbone", default=None,
                    help="Path to TrackerNano backbone ONNX (required with --tracker).")
    ap.add_argument("--nano-neckhead", default=None,
                    help="Path to TrackerNano neckhead ONNX (required with --tracker).")
    args = ap.parse_args()

    nano_backbone = None
    nano_neckhead = None
    if args.tracker:
        if not args.nano_backbone or not args.nano_neckhead:
            ap.error("--tracker requires both --nano-backbone and --nano-neckhead")
        nano_backbone = args.nano_backbone
        nano_neckhead = args.nano_neckhead

    create_debug_video(args.input, args.output, args.weights, args.world_model,
                       nano_backbone=nano_backbone, nano_neckhead=nano_neckhead)
