from pyexpat import features

import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque

# ==========================================
# 1. THE MODEL CLASS (Must match training exactly)
# ==========================================
class HoopsWorldModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, num_layers=2):
        super(HoopsWorldModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 3), 
            nn.Sigmoid()
        )
        
        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_size, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 2)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_summary = gru_out[:, -1, :]
        return self.classifier_head(final_summary), self.predictor_head(final_summary)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def safe_lm(landmarks, idx):
    if landmarks is None: return None
    lm = landmarks[idx]
    return float(lm.x), float(lm.y), float(lm.visibility)

def avg_xy(a, b):
    if a is None or b is None: return None
    return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5


def create_ball_tracker(weights_path, nano_backbone, nano_neckhead):
    """Create a cv2.BallTracker (YOLO + TrackerNano), same configuration as
    10_build_dribble_dataset.py / 30_generate_debug.py. The weights MUST be a
    format cv::dnn::readNet supports (e.g. _raw.onnx)."""
    params = cv2.BallTrackerParams()
    params.yoloModelCalibration = weights_path
    params.yoloModelDetection = weights_path
    params.yoloImgszCalibration = 320
    params.yoloImgszDetection = 320
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


# ==========================================
# 3. YOUTUBE URL HANDLING
# ==========================================
def resolve_video_source(video_source, cache_dir="../raw_dribbling_videos/youtube_cache"):
    """If video_source looks like a URL, download it with yt-dlp and return the local path."""
    if not (video_source.startswith("http://") or video_source.startswith("https://")):
        return video_source

    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is required to process YouTube URLs. Install it with: pip install yt-dlp"
        )

    os.makedirs(cache_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(cache_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_source, download=False)
        video_id = info["id"]
        local_path = os.path.join(cache_dir, f"{video_id}.mp4")

        if os.path.exists(local_path):
            print(f"✅ Using cached download: {local_path}")
        else:
            print(f"⬇️  Downloading YouTube video: {info.get('title', video_id)}")
            ydl.download([video_source])
            print(f"✅ Saved to: {local_path}")

    return local_path

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def run_inference(video_path, yolo_path, pytorch_path, output_path,
                  use_tracker=False, nano_backbone=None, nano_neckhead=None):
    video_path = resolve_video_source(video_path)

    print("Loading Models...")

    # Load ball detector — either stand-alone YOLO or the custom BallTracker
    # (YOLO + TrackerNano). When use_tracker is True, the YOLO weights MUST be
    # a format cv::dnn::readNet supports (e.g. _raw.onnx).
    yolo = None
    tracker = None
    if use_tracker:
        if not nano_backbone or not nano_neckhead:
            raise RuntimeError("use_tracker=True requires nano_backbone and nano_neckhead")
        print(f"Creating BallTracker (weights: {yolo_path}, "
              f"backbone: {nano_backbone}, neckhead: {nano_neckhead})")
        tracker = create_ball_tracker(yolo_path, nano_backbone, nano_neckhead)
    else:
        yolo = YOLO(yolo_path)

    # Load MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # Load Your Trained Brain
    model = HoopsWorldModel(input_size=26)
    model.load_state_dict(torch.load(pytorch_path, weights_only=True))
    model.eval() # Set to evaluation mode (turns off dropout)
    print("Models Loaded Successfully!")

    # ==========================================
    # BULLETPROOF VIDEO SETUP
    # ==========================================
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open {video_path}. Check the file name and folder path!")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if w == 0 or h == 0 or fps == 0:
        print("❌ ERROR: Video loaded, but dimensions are 0. The file might be corrupted.")
        return

    print(f"✅ Video loaded successfully: {w}x{h} at {fps} FPS")

    # Fallback Codec logic
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    except Exception as e:
        print("⚠️ mp4v codec failed, attempting fallback to avc1...")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # The 15-frame memory bank
    window = deque(maxlen=15)
    
    # Landmark indices
    LM = mp_pose.PoseLandmark
    idxs = {
        'LSH': LM.LEFT_SHOULDER.value, 'RSH': LM.RIGHT_SHOULDER.value,
        'LHIP': LM.LEFT_HIP.value, 'RHIP': LM.RIGHT_HIP.value,
        'LELB': LM.LEFT_ELBOW.value, 'RELB': LM.RIGHT_ELBOW.value,
        'LWR': LM.LEFT_WRIST.value, 'RWR': LM.RIGHT_WRIST.value,
        'LANK': LM.LEFT_ANKLE.value, 'RANK': LM.RIGHT_ANKLE.value
    }

    # The 15-frame memory bank
    window = deque(maxlen=15)
    
    # --- NEW: YOLO MEMORY ---
    last_seen_ball_norm = None    

    frame_count = 0
    pending_ghost = None  # prediction from previous frame, drawn on the current frame
    # --- UPGRADED COUNTING VARIABLES ---
    total_dribbles = 0
    is_dribbling = False  # This is our dynamic lock
    total_crossovers = 0
    is_crossing = False
    # Crossover lookahead: when a dribble triggers, open a CROSSOVER_LOOKAHEAD-frame
    # window and count a crossover if max(p_crossover) within the window exceeds
    # CROSSOVER_THRESHOLD. Decouples crossover from the exact dribble-apex frame —
    # the two probability peaks can be 1-2 frames apart.
    CROSSOVER_LOOKAHEAD = 5
    CROSSOVER_THRESHOLD = 0.30
    pending_crossover = None  # {frames_left, max_p} while a window is open
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # 1. Detect Ball
        ball_norm = None
        ball_detected_now = False  # True only if the detector found the ball this frame

        if tracker is not None:
            # Custom BallTracker (YOLO + TrackerNano). Internal state handles
            # occlusion recovery, so no need for last_seen_ball_norm fallback.
            result = tracker.processFrame(frame)
            if result.found:
                ball_detected_now = True
                # normBbox is (cx, cy, w, h) in [0,1] of the input frame
                cx_norm, cy_norm, _, _ = result.normBbox
                ball_norm = (float(cx_norm), float(cy_norm))
                last_seen_ball_norm = ball_norm

                # Draw bbox in absolute pixel coords
                bx, by, bw, bh = result.bbox
                # GREEN=tracker, RED=YOLO, ORANGE=calibrating
                if result.mode == cv2.BALL_TRACKER_MODE_TRACKER:
                    color = (0, 255, 0)
                elif result.mode == cv2.BALL_TRACKER_MODE_YOLO:
                    color = (0, 0, 255)
                elif result.mode == cv2.BALL_TRACKER_MODE_CALIBRATING:
                    color = (0, 165, 255)
                else:
                    color = (128, 128, 128)
                cv2.rectangle(frame, (int(bx), int(by)),
                              (int(bx + bw), int(by + bh)), color, 2)
            elif last_seen_ball_norm is not None:
                ball_norm = last_seen_ball_norm
        else:
            # Stand-alone YOLO path
            ball_res = yolo(frame, classes=[0], verbose=False)
            if ball_res and ball_res[0].boxes:
                box = ball_res[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                ball_norm = ((x1+x2)/2 / w, (y1+y2)/2 / h)
                ball_detected_now = True

                # Save the coordinates to memory!
                last_seen_ball_norm = ball_norm

                # Draw YOLO Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            elif last_seen_ball_norm is not None:
                # REAL-TIME OCCLUSION RECOVERY
                # If YOLO goes blind due to motion blur, use the last known location
                # so the velocity and distance math doesn't spike to 0.0!
                ball_norm = last_seen_ball_norm

        # 2. Detect Body
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        lms = pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
        
        if pose_res.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 3. Build the 26-Feature Vector (index 25 is Ball_Detected)
        features = [0.0] * 26
        features[25] = 1.0 if ball_detected_now else 0.0

        hip_center = None
        if lms:
            lhip, rhip = safe_lm(lms, idxs['LHIP']), safe_lm(lms, idxs['RHIP'])
            hip_center = avg_xy(lhip, rhip)

            if hip_center:
                hcx, hcy = hip_center
                
                def rel(lm): return (lm[0]-hcx, lm[1]-hcy, lm[2]) if lm else (0.0, 0.0, 0.0)

                if ball_norm:
                    features[0], features[1] = ball_norm[0]-hcx, ball_norm[1]-hcy
                
                features[2:5] = rel(safe_lm(lms, idxs['LELB']))
                features[5:8] = rel(safe_lm(lms, idxs['RELB']))
                
                # We need to save the wrists to variables so we can calculate distance!
                l_wrist = rel(safe_lm(lms, idxs['LWR']))
                r_wrist = rel(safe_lm(lms, idxs['RWR']))
                
                features[8:11] = l_wrist
                features[11:14] = r_wrist
                features[14:17] = rel(safe_lm(lms, idxs['LANK']))
                features[17:20] = rel(safe_lm(lms, idxs['RANK']))
                
                lsh, rsh = safe_lm(lms, idxs['LSH']), safe_lm(lms, idxs['RSH'])
                sh_center = avg_xy(lsh, rsh)
                if sh_center: features[20] = abs(sh_center[1] - hcy)
                
                # --- NEW MATH FEATURES ---
                # Dist_Ball_L_Wrist
                features[21] = np.sqrt((features[0] - l_wrist[0])**2 + (features[1] - l_wrist[1])**2)
                # Dist_Ball_R_Wrist
                features[22] = np.sqrt((features[0] - r_wrist[0])**2 + (features[1] - r_wrist[1])**2)
                
                # Delta_Ball_Y (Current Y - Previous frame's Y from the memory bank)
                if len(window) > 0:
                    previous_y = window[-1][1] # Get Rel_Ball_Y from the last frame in the deque
                    features[23] = features[1] - previous_y
                else:
                    features[23] = 0.0 # First frame has no velocity
                # Delta_Ball_X (Current X - Previous frame's X from the memory bank)
                if len(window) > 0:
                    previous_x = window[-1][0] # Get Rel_Ball_X from the last frame in the deque
                    features[24] = features[0] - previous_x
                else:
                    features[24] = 0.0 # First frame has no velocity

        # 4. Add to Memory Bank
        window.append(features)

        # 5. Run AI Inference (Only if we have a full 15 frames)
        new_ghost = None
        if len(window) == 15:
            x_seq = torch.FloatTensor(np.array(window)).unsqueeze(0)

            with torch.no_grad():
                pred_actions, pred_coords = model(x_seq)

            # Extract numbers from tensors
            probs = pred_actions[0].numpy()
            p_dribble, p_crossover, p_touch = probs[0], probs[1], probs[2]
            
            # --- HYSTERESIS STATE MACHINE ---
            # 1. Trigger: If confident and the lock is OFF, count it and engage the lock!
            #    Open a crossover-lookahead window seeded with the current p_crossover.
            #    If a previous window is still open (fast-cadence dribbles overlap),
            #    finalize it first so its crossover decision isn't lost.
            if p_dribble > 0.50 and not is_dribbling:
                total_dribbles += 1
                is_dribbling = True
                if pending_crossover is not None and pending_crossover['max_p'] > CROSSOVER_THRESHOLD:
                    total_crossovers += 1
                pending_crossover = {
                    'frames_left': CROSSOVER_LOOKAHEAD,
                    'max_p': p_crossover,
                }

            # 2. Reset: If the probability drops below our safety threshold, remove the lock!
            elif p_dribble < 0.30 and is_dribbling:
                is_dribbling = False

            # 3. Crossover lookahead: while a window is open, track max p_crossover.
            #    When the window closes, count it if the max crossed CROSSOVER_THRESHOLD.
            if pending_crossover is not None:
                pending_crossover['max_p'] = max(pending_crossover['max_p'], p_crossover)
                pending_crossover['frames_left'] -= 1
                if pending_crossover['frames_left'] <= 0:
                    if pending_crossover['max_p'] > CROSSOVER_THRESHOLD:
                        total_crossovers += 1
                    pending_crossover = None

            # --- CROSSOVER HYSTERESIS ---
            #if p_crossover > 0.70 and not is_crossing:
            #    total_crossovers += 1
            #    is_crossing = True
            #elif p_crossover < 0.30 and is_crossing:
            #    is_crossing = False

            next_x_rel, next_y_rel = pred_coords[0].numpy()

            # --- UI DRAWING ---
            cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)

            color_d = (0, 255, 0) if p_dribble > 0.5 else (255, 255, 255)
            cv2.putText(frame, f"Dribble:   {p_dribble*100:.1f}%", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_d, 2)
            print(f"Dribble Probability: {p_dribble:.4f}")  # Debug print for dribble
            color_c = (0, 0, 255) if p_crossover > 0.5 else (255, 255, 255)
            cv2.putText(frame, f"Crossover: {p_crossover*100:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_c, 2)
            print(f"Crossover Probability: {p_crossover:.4f}")  # Debug print for crossover
            cv2.putText(frame, f"Hand Touch: {p_touch*100:.1f}%", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            print(f"Hand Touch Probability: {p_touch:.4f}")  # Debug print for hand touch
            # --- NEW: DRAW THE SCOREBOARD ---
            cv2.putText(frame, f"TOTAL DRIBBLES: {total_dribbles}", (20, 155), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"TOTAL CROSSOVERS: {total_crossovers}", (20, 190), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
            
            if hip_center:
                new_ghost = (
                    int((next_x_rel + hip_center[0]) * w),
                    int((next_y_rel + hip_center[1]) * h),
                )

        # Draw the prediction made on the PREVIOUS frame onto this frame
        if pending_ghost is not None:
            gx, gy = pending_ghost
            cv2.circle(frame, (gx, gy), 15, (255, 0, 255), -1)
            cv2.putText(frame, "AI Target", (gx + 20, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        pending_ghost = new_ghost

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Success! Check out your AI in action: {output_path}")

if __name__ == "__main__":
    # --- CHANGE THESE PATHS TO MATCH YOUR FILES ---
    # TEST_VIDEO can be a local path OR a YouTube URL (requires: pip install yt-dlp)
    TEST_VIDEO = "../raw_dribbling_videos/crossover_02_05_1.mp4"
    # TEST_VIDEO = "https://www.youtube.com/watch?v=oADaM2L1YLc"
    # TEST_VIDEO = "https://www.youtube.com/shorts/EK34qP2XWFM"

    # When USE_TRACKER is True, YOLO_WEIGHTS must be a format
    # cv::dnn::readNet supports (e.g. _raw.onnx). When False, any Ultralytics
    # weights file works (.pt, .onnx, etc.).
    USE_TRACKER = True
    YOLO_WEIGHTS = "../models/ball_detection_v26n_320_07_04_raw.onnx" if USE_TRACKER \
        else "../models/ball_detection_v26n_640_07_04.pt"
    NANO_BACKBONE = "../models/nanotrack_backbone_sim_v2.onnx"
    NANO_NECKHEAD = "../models/nanotrack_head_sim_v2.onnx"

    PYTORCH_WEIGHTS = "../models/hoops_world_model_best.pth"
    OUTPUT_VIDEO = "../raw_dribbling_videos/debug_videos/ai_crossover_02_05_1.mp4"

    run_inference(
        TEST_VIDEO, YOLO_WEIGHTS, PYTORCH_WEIGHTS, OUTPUT_VIDEO,
        use_tracker=USE_TRACKER,
        nano_backbone=NANO_BACKBONE if USE_TRACKER else None,
        nano_neckhead=NANO_NECKHEAD if USE_TRACKER else None,
    )