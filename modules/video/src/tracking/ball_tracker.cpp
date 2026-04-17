// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// BallTracker: High-level basketball detection + tracking pipeline
// Combines YOLO (via cv::dnn) with TrackerNano for robust ball tracking.
// Uses a Dynamic Color Model (EMA histogram) for drift detection.

#include "../precomp.hpp"
#include <deque>
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

// ============================================================================
// BallTracker base
// ============================================================================

BallTracker::BallTracker() {}
BallTracker::~BallTracker() {}

BallTrackerParams::BallTrackerParams()
{
    // YOLO
    yoloModelCalibration = "";
    yoloModelDetection = "";
    yoloImgszCalibration = 640;
    yoloImgszDetection = 640;
    yoloConfidence = 0.5f;
    yoloClassId = 0;

    // TrackerNano
    nanoBackbone = "";
    nanoNeckhead = "";
    searchCrops = 5;
    earlyExitScore = 0.85f;
    motionHistory = 5;

    // Tracking
    confidenceThreshold = 0.25f;

    // Calibration
    numTemplates = 5;
    calibrationFrames = 100;

    // Periodic
    yoloPeriodic = 10;
    redetectInterval = 0;

    // Sanity
    maxBboxArea = 0.15f;
    maxBboxJump = 4.0f;
    maxAspect = 3.0f;

    // Dynamic Color Model
    colorAlpha = 0.10f;
    driftThreshold = 0.30f;
    agreementDistance = 50.0f;
    maxSizeChangeRatio = 1.8f;
    sizeHistoryLength = 10;
    maxRecoveryFrames = 30;
    noColorValidation = false;

    // DNN
#ifdef HAVE_OPENCV_DNN
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    backend = -1;
    target = -1;
#endif
}

BallTrackerResult::BallTrackerResult()
    : found(false), mode(BALL_TRACKER_MODE_LOST), confidence(-1.f), frameNumber(0)
{
}

#ifdef HAVE_OPENCV_DNN

// ============================================================================
// DynamicColorModel -EMA histogram-based appearance validation
// ============================================================================

class DynamicColorModel
{
public:
    static const int H_BINS = 18;
    static const int S_BINS = 16;
    static const int COMPARE_W = 32;
    static const int COMPARE_H = 32;

    DynamicColorModel(float alpha, float driftThreshold)
        : alpha_(alpha), driftThreshold_(driftThreshold), initialized_(false), updateCount_(0) {}

    // Color model is ready for validation only after multiple YOLO confirmations
    bool isReady() const { return initialized_ && updateCount_ >= 3; }

    // Called ONLY when we are confident this is the ball (YOLO confirmed)
    void update(const Mat& frame, const Rect& bbox)
    {
        Mat crop = safeCrop(frame, bbox);
        if (crop.empty()) return;
        Mat cropHist = computeHist(crop);

        updateCount_++;
        if (!initialized_)
        {
            runningHist_ = cropHist.clone();
            anchorHist_ = cropHist.clone();
            initialized_ = true;
        }
        else
        {
            // EMA blend: slowly adapt to motion blur, lighting changes
            runningHist_ = (1.0f - alpha_) * runningHist_ + alpha_ * cropHist;
            normalize(runningHist_, runningHist_, 0, 1, NORM_MINMAX);
        }
    }

    // Returns similarity score (0-1). Higher = more similar to known ball.
    float similarity(const Mat& frame, const Rect& bbox) const
    {
        if (!initialized_) return 1.0f;  // No model yet -accept everything
        Mat crop = safeCrop(frame, bbox);
        if (crop.empty()) return 0.0f;
        Mat cropHist = computeHist(crop);

        float runningScore = (float)compareHist(cropHist, runningHist_, HISTCMP_CORREL);
        float anchorScore = (float)compareHist(cropHist, anchorHist_, HISTCMP_CORREL);
        // Use the higher of the two -anchor catches cases where
        // the running model drifted but the ball returned to original appearance
        return std::max(runningScore, anchorScore);
    }

    // Check if a bbox looks like the ball
    bool validate(const Mat& frame, const Rect& bbox) const
    {
        return similarity(frame, bbox) >= driftThreshold_;
    }

private:
    Mat safeCrop(const Mat& frame, const Rect& bbox) const
    {
        Rect clipped = bbox & Rect(0, 0, frame.cols, frame.rows);
        if (clipped.width <= 0 || clipped.height <= 0) return Mat();
        return frame(clipped);
    }

    Mat computeHist(const Mat& crop) const
    {
        Mat resized, hsv;
        cv::resize(crop, resized, Size(COMPARE_W, COMPARE_H));
        cvtColor(resized, hsv, COLOR_BGR2HSV);

        int channels[] = {0, 1};
        int histSize[] = {H_BINS, S_BINS};
        float hRange[] = {0, 180};
        float sRange[] = {0, 256};
        const float* ranges[] = {hRange, sRange};

        Mat hist;
        calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges);
        normalize(hist, hist, 0, 1, NORM_MINMAX);
        return hist;
    }

    float alpha_;
    float driftThreshold_;
    bool initialized_;
    int updateCount_;
    Mat runningHist_;    // EMA histogram -evolves over time
    Mat anchorHist_;     // Frozen histogram from first YOLO confirmation
};


// ============================================================================
// SizeTracker -rolling window bbox size consistency check
// ============================================================================

class SizeTracker
{
public:
    SizeTracker(int windowSize, float maxChangeRatio)
        : windowSize_(windowSize), maxChangeRatio_(maxChangeRatio) {}

    void push(const Rect& bbox)
    {
        float diag = std::sqrt((float)(bbox.width * bbox.width + bbox.height * bbox.height));
        history_.push_back(diag);
        if ((int)history_.size() > windowSize_)
            history_.pop_front();
    }

    // Returns true if the bbox size is consistent with recent history
    bool isConsistent(const Rect& bbox) const
    {
        if (history_.size() < 3) return true;  // Not enough data
        float diag = std::sqrt((float)(bbox.width * bbox.width + bbox.height * bbox.height));
        float avg = averageDiag();
        if (avg <= 0) return true;
        float ratio = diag / avg;
        return ratio <= maxChangeRatio_ && ratio >= (1.0f / maxChangeRatio_);
    }

    float averageDiag() const
    {
        if (history_.empty()) return 0.f;
        float sum = 0.f;
        for (float d : history_) sum += d;
        return sum / (float)history_.size();
    }

    void clear() { history_.clear(); }

private:
    int windowSize_;
    float maxChangeRatio_;
    std::deque<float> history_;
};


// ============================================================================
// YOLO detector using cv::dnn
// ============================================================================

class YoloDetector
{
public:
    YoloDetector() : loaded_(false), imgSize_(640), confThreshold_(0.5f), classId_(0) {}

    bool load(const std::string& modelPath, int imgSize, float confThreshold, int classId,
              int backend, int target)
    {
        if (modelPath.empty()) return false;
        net_ = dnn::readNet(modelPath);
        if (net_.empty()) return false;
        net_.setPreferableBackend(backend);
        net_.setPreferableTarget(target);
        imgSize_ = imgSize;
        confThreshold_ = confThreshold;
        classId_ = classId;
        loaded_ = true;
        return true;
    }

    bool isLoaded() const { return loaded_; }

    Rect detect(const Mat& frame, float& outConf) const
    {
        outConf = 0.f;
        if (!loaded_) return Rect();

        // Letterbox preprocessing
        int origW = frame.cols;
        int origH = frame.rows;
        float scale = std::min((float)imgSize_ / origW, (float)imgSize_ / origH);
        int newW = (int)(origW * scale);
        int newH = (int)(origH * scale);
        int padX = (imgSize_ - newW) / 2;
        int padY = (imgSize_ - newH) / 2;

        Mat resized;
        cv::resize(frame, resized, Size(newW, newH));
        Mat letterboxed(imgSize_, imgSize_, CV_8UC3, Scalar(114, 114, 114));
        resized.copyTo(letterboxed(Rect(padX, padY, newW, newH)));

        Mat blob = dnn::blobFromImage(letterboxed, 1.0 / 255.0, Size(imgSize_, imgSize_),
                                      Scalar(), true, false);
        net_.setInput(blob);

        std::vector<Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty()) return Rect();

        Mat out = outputs[0];

        if (out.dims == 3)
        {
            int d1 = out.size[1];
            int d2 = out.size[2];
            if (d1 < d2)
            {
                out = out.reshape(1, d1);
                cv::transpose(out, out);
            }
            else
            {
                out = out.reshape(1, d1);
            }
        }

        int rows = out.rows;
        int cols = out.cols;
        if (rows == 0 || cols < 5) return Rect();

        Rect bestBox;
        float bestConf = 0.f;
        float invScale = 1.0f / scale;

        int scoreCol = 4;
        bool hasClassIdCol = false;

        if (cols == 6)
        {
            hasClassIdCol = true;
        }
        else if (cols == 5)
        {
            scoreCol = 4;
        }
        else if (cols > 5)
        {
            scoreCol = 4 + classId_;
            if (scoreCol >= cols) return Rect();
        }

        for (int i = 0; i < rows; i++)
        {
            const float* row = out.ptr<float>(i);

            if (hasClassIdCol)
            {
                int clsId = (int)row[5];
                if (clsId != classId_) continue;
            }

            float conf = row[scoreCol];
            if (conf < confThreshold_ || conf <= bestConf) continue;

            float x1 = (row[0] - padX) * invScale;
            float y1 = (row[1] - padY) * invScale;
            float x2 = (row[2] - padX) * invScale;
            float y2 = (row[3] - padY) * invScale;

            bestBox = Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            bestConf = conf;
        }

        outConf = bestConf;
        return bestBox;
    }

private:
    mutable dnn::Net net_;
    bool loaded_;
    int imgSize_;
    float confThreshold_;
    int classId_;
};


// ============================================================================
// Sanity check helper
// ============================================================================

static bool isBboxSane(const Rect& bbox, int frameW, int frameH,
                       const Rect& prevBbox, bool hasPrev,
                       float maxAreaRatio, float maxJumpFactor, float maxAspect,
                       std::string& reason)
{
    reason.clear();
    if (bbox.width <= 0 || bbox.height <= 0)
    {
        reason = "zero-size bbox";
        return false;
    }

    float frameArea = (float)(frameW * frameH);
    float bboxArea = (float)(bbox.width * bbox.height);
    float areaRatio = bboxArea / frameArea;

    if (areaRatio > maxAreaRatio)
    {
        reason = "bbox too large";
        return false;
    }

    float aspect = (float)bbox.width / bbox.height;
    if (aspect > maxAspect || aspect < 1.0f / maxAspect)
    {
        reason = "bad aspect ratio";
        return false;
    }

    if (hasPrev && prevBbox.width > 0 && prevBbox.height > 0)
    {
        float prevArea = (float)(prevBbox.width * prevBbox.height);
        if (prevArea > 0 && bboxArea > prevArea * maxJumpFactor)
        {
            reason = "bbox grew too fast";
            return false;
        }
    }

    return true;
}

// Helper: center distance between two Rects
static float centerDistance(const Rect& a, const Rect& b)
{
    float ax = a.x + a.width * 0.5f;
    float ay = a.y + a.height * 0.5f;
    float bx = b.x + b.width * 0.5f;
    float by = b.y + b.height * 0.5f;
    return std::sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

// Helper: fill normBbox from pixel bbox and frame dims
static void fillNormBbox(BallTrackerResult& result, int frameW, int frameH)
{
    result.normBbox = Rect2d(
        (result.bbox.x + result.bbox.width * 0.5) / frameW,
        (result.bbox.y + result.bbox.height * 0.5) / frameH,
        (double)result.bbox.width / frameW,
        (double)result.bbox.height / frameH
    );
}


// ============================================================================
// BallTrackerImpl
// ============================================================================

class BallTrackerImpl : public BallTracker
{
public:
    BallTrackerImpl(const BallTrackerParams& params)
        : params_(params)
        , colorModel_(params.colorAlpha, params.driftThreshold)
        , sizeTracker_(params.sizeHistoryLength, params.maxSizeChangeRatio)
        , frameNum_(0)
        , calibrating_(false)
        , calibDone_(false)
        , trackingActive_(false)
        , templatesCollected_(0)
        , calibFrameCount_(0)
        , trackerConf_(-1.f)
        , recoveryFrames_(0)
        , inRecovery_(false)
    {
        // Load YOLO models
        yoloCalib_.load(params.yoloModelCalibration, params.yoloImgszCalibration,
                        params.yoloConfidence, params.yoloClassId,
                        params.backend, params.target);

        yoloDetect_.load(params.yoloModelDetection, params.yoloImgszDetection,
                         params.yoloConfidence, params.yoloClassId,
                         params.backend, params.target);

        // Create TrackerNano if models are provided
        useTracker_ = !params.nanoBackbone.empty() && !params.nanoNeckhead.empty();
        if (useTracker_)
        {
            TrackerNano::Params nanoParams;
            nanoParams.backbone = params.nanoBackbone;
            nanoParams.neckhead = params.nanoNeckhead;
            nanoParams.backend = params.backend;
            nanoParams.target = params.target;
            nanoParams.searchCrops = params.searchCrops;
            nanoParams.earlyExitScore = params.earlyExitScore;
            nanoParams.motionHistory = params.motionHistory;
            tracker_ = TrackerNano::create(nanoParams);
            calibrating_ = true;
        }
    }

    BallTrackerResult processFrame(InputArray _frame) CV_OVERRIDE
    {
        Mat frame = _frame.getMat();
        CV_Assert(!frame.empty());

        frameNum_++;
        int frameW = frame.cols;
        int frameH = frame.rows;

        BallTrackerResult result;
        result.frameNumber = frameNum_;
        result.found = false;
        result.mode = BALL_TRACKER_MODE_LOST;
        result.confidence = -1.f;

        // ==============================================================
        // Calibration phase
        // ==============================================================
        if (calibrating_)
        {
            calibFrameCount_++;

            Rect detBbox;
            float detConf = 0.f;
            if (yoloCalib_.isLoaded())
            {
                detBbox = yoloCalib_.detect(frame, detConf);
            }

            if (detBbox.width > 0 && detBbox.height > 0)
            {
                std::string reason;
                bool sane = isBboxSane(detBbox, frameW, frameH,
                                       lastBbox_, hasLastBbox_,
                                       params_.maxBboxArea, params_.maxBboxJump, params_.maxAspect,
                                       reason);
                if (sane)
                {
                    tracker_->addTemplate(frame, detBbox);
                    colorModel_.update(frame, detBbox);
                    sizeTracker_.push(detBbox);
                    templatesCollected_++;
                    lastBbox_ = detBbox;
                    hasLastBbox_ = true;

                    result.bbox = detBbox;
                    result.found = true;
                    result.confidence = detConf;
                    printf("[BallTracker] Calibration frame %d: template %d collected, bbox=(%d,%d,%dx%d) conf=%.4f\n",
                           frameNum_, templatesCollected_, detBbox.x, detBbox.y, detBbox.width, detBbox.height, detConf);
                }
                else
                {
                    printf("[BallTracker] Calibration frame %d: YOLO bbox rejected: %s\n",
                           frameNum_, reason.c_str());
                }
            }

            result.mode = BALL_TRACKER_MODE_CALIBRATING;

            bool done = (templatesCollected_ >= params_.numTemplates) ||
                        (calibFrameCount_ >= params_.calibrationFrames);

            if (done && templatesCollected_ > 0)
            {
                tracker_->finalizeTemplates();
                tracker_->init(frame, lastBbox_);
                trackingActive_ = true;
                calibrating_ = false;
                calibDone_ = true;
                printf("[BallTracker] Calibration complete: %d template(s) in %d frames\n",
                       templatesCollected_, calibFrameCount_);
            }
            else if (done && templatesCollected_ == 0)
            {
                printf("[BallTracker] Calibration FAILED: no ball detected in %d frames\n",
                       calibFrameCount_);
                calibrating_ = false;
                calibDone_ = false;
                useTracker_ = false;
            }

            if (result.found)
                fillNormBbox(result, frameW, frameH);
            return result;
        }

        // ==============================================================
        // Should YOLO run this frame?
        // ==============================================================
        bool runYolo = !trackingActive_
            || (params_.yoloPeriodic > 0 && frameNum_ % params_.yoloPeriodic == 0)
            || (params_.redetectInterval > 0 && frameNum_ % params_.redetectInterval == 0)
            || inRecovery_;

        if (!useTracker_)
            runYolo = true;

        // ==============================================================
        // Phase 1: TrackerNano update
        // ==============================================================
        bool trackerDrifted = false;

        if (trackingActive_ && tracker_ && useTracker_)
        {
            Rect trkBbox;
            bool success = tracker_->update(frame, trkBbox);
            trackerConf_ = tracker_->getTrackingScore();

            std::string reason;
            bool sane = isBboxSane(trkBbox, frameW, frameH,
                                   lastBbox_, hasLastBbox_,
                                   params_.maxBboxArea, params_.maxBboxJump, params_.maxAspect,
                                   reason);

            // --- Drift detection: color validation ---
            bool colorOk = true;
            float colorScore = 1.0f;
            if (!params_.noColorValidation && sane && colorModel_.isReady())
            {
                colorScore = colorModel_.similarity(frame, trkBbox);
                colorOk = (colorScore >= params_.driftThreshold);
            }

            // --- Drift detection: size consistency ---
            bool sizeOk = true;
            if (sane)
            {
                sizeOk = sizeTracker_.isConsistent(trkBbox);
            }

            if (success && sane && colorOk && sizeOk)
            {
                lastBbox_ = trkBbox;
                hasLastBbox_ = true;
                result.bbox = trkBbox;
                result.found = true;
                result.mode = BALL_TRACKER_MODE_TRACKER;
                result.confidence = trackerConf_;
                sizeTracker_.push(trkBbox);

                // Exit recovery if we were in it
                if (inRecovery_)
                {
                    inRecovery_ = false;
                    recoveryFrames_ = 0;
                }
            }
            else
            {
                // Tracker drifted or lost
                trackerDrifted = true;
                trackingActive_ = false;
                runYolo = true;
                // Clear lastBbox so sanity checks don't compare against drifted position
                hasLastBbox_ = false;

                if (!colorOk)
                    printf("[BallTracker] Frame %d: DRIFT detected (color=%.3f < %.3f)\n",
                           frameNum_, colorScore, params_.driftThreshold);
                else if (!sizeOk)
                    printf("[BallTracker] Frame %d: DRIFT detected (size inconsistent, avg=%.0f)\n",
                           frameNum_, sizeTracker_.averageDiag());
                else if (!sane)
                    printf("[BallTracker] Frame %d: tracker rejected (%s)\n",
                           frameNum_, reason.c_str());
            }
        }

        // ==============================================================
        // Phase 2: YOLO detection
        // ==============================================================
        if (runYolo)
        {
            Rect detBbox;
            float detConf = 0.f;
            if (yoloDetect_.isLoaded())
            {
                detBbox = yoloDetect_.detect(frame, detConf);
            }

            if (detBbox.width > 0 && detBbox.height > 0)
            {
                std::string reason;
                bool sane = isBboxSane(detBbox, frameW, frameH,
                                       lastBbox_, hasLastBbox_,
                                       params_.maxBboxArea, params_.maxBboxJump, params_.maxAspect,
                                       reason);
                if (!sane)
                {
                    printf("[BallTracker] Frame %d: YOLO bbox rejected (%s)\n", frameNum_, reason.c_str());
                    detBbox = Rect();
                }
            }
            else if (inRecovery_)
            {
                printf("[BallTracker] Frame %d: RECOVERY -YOLO found nothing\n", frameNum_);
            }

            if (detBbox.width > 0 && detBbox.height > 0)
            {
                // --- YOLO Confirmation Gate ---
                // If tracker already found the ball this frame (periodic check),
                // compare tracker vs YOLO positions before deciding
                if (result.found && result.mode == BALL_TRACKER_MODE_TRACKER)
                {
                    float dist = centerDistance(result.bbox, detBbox);

                    if (dist <= params_.agreementDistance)
                    {
                        // They agree -tracker is healthy, keep tracker result (smoother)
                        // But update color model since YOLO confirmed
                        colorModel_.update(frame, result.bbox);
                        // Don't overwrite result -keep tracker
                    }
                    else
                    {
                        // They disagree -but only trust YOLO if it passes color validation
                        // (prevents re-init onto player/head when YOLO misdetects)
                        float yoloColorScore = colorModel_.isReady()
                            ? colorModel_.similarity(frame, detBbox) : 1.0f;
                        bool yoloLooksLikeBall = params_.noColorValidation
                            || yoloColorScore >= params_.driftThreshold;

                        if (yoloLooksLikeBall)
                        {
                            printf("[BallTracker] Frame %d: tracker/YOLO disagree (dist=%.0f) -YOLO color=%.3f OK, re-init\n",
                                   frameNum_, dist, yoloColorScore);

                            colorModel_.update(frame, detBbox);
                            sizeTracker_.push(detBbox);
                            lastBbox_ = detBbox;
                            hasLastBbox_ = true;
                            result.bbox = detBbox;
                            result.found = true;
                            result.mode = BALL_TRACKER_MODE_YOLO;
                            result.confidence = detConf;

                            if (useTracker_ && calibDone_ && tracker_)
                            {
                                tracker_->init(frame, detBbox);
                                trackingActive_ = true;
                            }
                        }
                        else
                        {
                            // YOLO detection doesn't look like the ball either -both are wrong
                            printf("[BallTracker] Frame %d: tracker/YOLO disagree (dist=%.0f) -YOLO color=%.3f REJECTED, entering recovery\n",
                                   frameNum_, dist, yoloColorScore);
                            trackingActive_ = false;
                            trackerDrifted = true;
                            result.found = false;
                        }
                    }
                }
                else
                {
                    // Tracker didn't find it (or wasn't running) -use YOLO result
                    // Only validate YOLO against color model when:
                    //   - tracker is active (not YOLO-only fallback)
                    //   - color model is initialized
                    //   - not in recovery or drift (color model may be stale)
                    bool yoloColorOk = true;
                    if (!params_.noColorValidation && useTracker_ && colorModel_.isReady()
                        && !inRecovery_ && !trackerDrifted)
                    {
                        yoloColorOk = colorModel_.validate(frame, detBbox);
                    }

                    if (yoloColorOk)
                    {
                        colorModel_.update(frame, detBbox);
                        sizeTracker_.push(detBbox);
                        lastBbox_ = detBbox;
                        hasLastBbox_ = true;
                        result.bbox = detBbox;
                        result.found = true;
                        result.mode = BALL_TRACKER_MODE_YOLO;
                        result.confidence = detConf;

                        if (useTracker_ && calibDone_ && tracker_)
                        {
                            tracker_->init(frame, detBbox);
                            trackingActive_ = true;
                        }

                        // Exit recovery
                        inRecovery_ = false;
                        recoveryFrames_ = 0;
                    }
                    else
                    {
                        printf("[BallTracker] Frame %d: YOLO detection rejected by color model\n",
                               frameNum_);
                    }
                }
            }
        }

        // ==============================================================
        // Recovery mode
        // ==============================================================
        if (!result.found)
        {
            if (!inRecovery_ && trackerDrifted)
            {
                // Just entered recovery
                inRecovery_ = true;
                recoveryFrames_ = 1;
                printf("[BallTracker] Frame %d: entering RECOVERY mode\n", frameNum_);
            }
            else if (inRecovery_)
            {
                recoveryFrames_++;
                if (recoveryFrames_ > params_.maxRecoveryFrames)
                {
                    // Give up recovery -stop running YOLO every frame
                    printf("[BallTracker] Frame %d: recovery exhausted (%d frames)\n",
                           frameNum_, recoveryFrames_);
                    inRecovery_ = false;
                    recoveryFrames_ = 0;
                    sizeTracker_.clear();
                }
            }

            result.mode = BALL_TRACKER_MODE_LOST;
            trackingActive_ = false;
        }
        else
        {
            fillNormBbox(result, frameW, frameH);
        }

        return result;
    }

    bool isCalibrating() const CV_OVERRIDE { return calibrating_; }
    int getFrameCount() const CV_OVERRIDE { return frameNum_; }
    int getTemplatesCollected() const CV_OVERRIDE { return templatesCollected_; }

private:
    BallTrackerParams params_;
    DynamicColorModel colorModel_;
    SizeTracker sizeTracker_;

    YoloDetector yoloCalib_;
    YoloDetector yoloDetect_;

    Ptr<TrackerNano> tracker_;
    bool useTracker_;

    int frameNum_;
    bool calibrating_;
    bool calibDone_;
    bool trackingActive_;
    int templatesCollected_;
    int calibFrameCount_;
    float trackerConf_;

    // Recovery state
    int recoveryFrames_;
    bool inRecovery_;

    Rect lastBbox_;
    bool hasLastBbox_ = false;
};

Ptr<BallTracker> BallTracker::create(const BallTrackerParams& params)
{
    return makePtr<BallTrackerImpl>(params);
}

#else  // !HAVE_OPENCV_DNN

Ptr<BallTracker> BallTracker::create(const BallTrackerParams& /*params*/)
{
    CV_Error(Error::StsNotImplemented, "BallTracker requires opencv_dnn module");
}

#endif  // HAVE_OPENCV_DNN

}  // namespace cv
