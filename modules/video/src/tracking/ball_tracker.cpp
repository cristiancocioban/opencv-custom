// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// BallTracker: High-level basketball detection + tracking pipeline
// Combines YOLO (via cv::dnn) with TrackerNano for robust ball tracking.

#include "../precomp.hpp"
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

    // Template bank
    templateBankSize = 8;
    templateSimilarity = 0.40f;
    noTemplateValidation = false;

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
// TemplateBank — HSV histogram-based appearance validation
// ============================================================================

class TemplateBank
{
public:
    static const int H_BINS = 18;
    static const int S_BINS = 16;
    static const int COMPARE_W = 32;
    static const int COMPARE_H = 32;

    TemplateBank(int maxSize, float minSimilarity)
        : maxSize_(maxSize), minSimilarity_(minSimilarity) {}

    bool isFull() const { return (int)entries_.size() >= maxSize_; }
    bool isEmpty() const { return entries_.empty(); }
    int size() const { return (int)entries_.size(); }

    bool add(const Mat& frame, const Rect& bbox)
    {
        if (isFull()) return false;
        Mat crop = safeCrop(frame, bbox);
        if (crop.empty()) return false;
        entries_.push_back(computeHist(crop));
        return true;
    }

    // Returns (accepted, similarity)
    std::pair<bool, float> validate(const Mat& frame, const Rect& bbox) const
    {
        float score = bestSimilarity(frame, bbox);
        return std::make_pair(score >= minSimilarity_, score);
    }

    float bestSimilarity(const Mat& frame, const Rect& bbox) const
    {
        if (isEmpty()) return 1.0f;
        Mat crop = safeCrop(frame, bbox);
        if (crop.empty()) return 0.0f;
        Mat candHist = computeHist(crop);
        float best = -1.0f;
        for (const auto& h : entries_)
        {
            float score = (float)compareHist(candHist, h, HISTCMP_CORREL);
            if (score > best) best = score;
        }
        return best;
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

    int maxSize_;
    float minSimilarity_;
    std::vector<Mat> entries_;
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

    // Returns best detection as Rect, or empty Rect if nothing found
    // Also sets outConf to the confidence of the best detection
    Rect detect(const Mat& frame, float& outConf) const
    {
        outConf = 0.f;
        if (!loaded_) return Rect();

        // Letterbox preprocessing: resize preserving aspect ratio and pad with gray
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

        // Forward
        std::vector<Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        if (outputs.empty()) return Rect();

        // Parse YOLO output: shape [1, N, 5+C] or [1, 5+C, N] (YOLOv8/v11 transposed)
        Mat out = outputs[0];

        // Handle 3D output: [1, rows, cols]
        if (out.dims == 3)
        {
            int d1 = out.size[1];
            int d2 = out.size[2];
            // YOLOv8/v11 outputs [1, 5+C, N] — need to transpose to [1, N, 5+C]
            if (d1 < d2)
            {
                out = out.reshape(1, d1);  // [5+C, N]
                cv::transpose(out, out);   // [N, 5+C]
            }
            else
            {
                out = out.reshape(1, d1);  // [N, 5+C]
            }
        }

        int rows = out.rows;
        int cols = out.cols;
        if (rows == 0 || cols < 5) return Rect();

        Rect bestBox;
        float bestConf = 0.f;

        // Letterbox-to-original coordinate transform
        float invScale = 1.0f / scale;

        // Output format after stripping NMS (recommended):
        //   [N, 5]: [x1, y1, x2, y2, score] in letterbox coords (single-class)
        //   [N, 4+C]: [x1, y1, x2, y2, cls0_score, cls1_score, ...] (multi-class)
        // Also supports NMS format:
        //   [N, 6]: [x1, y1, x2, y2, score, class_id]

        // All coordinates are x1,y1,x2,y2 in letterbox space
        int scoreCol = 4;  // confidence/score column
        bool hasClassIdCol = false;

        if (cols == 6)
        {
            // NMS format with class_id column: [x1,y1,x2,y2,score,class_id]
            hasClassIdCol = true;
        }
        else if (cols == 5)
        {
            // Single-class: [x1,y1,x2,y2,score]
            scoreCol = 4;
        }
        else if (cols > 5)
        {
            // Multi-class: [x1,y1,x2,y2,cls0,cls1,...]
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


// ============================================================================
// BallTrackerImpl
// ============================================================================

class BallTrackerImpl : public BallTracker
{
public:
    BallTrackerImpl(const BallTrackerParams& params)
        : params_(params)
        , bank_(params.templateBankSize, params.templateSimilarity)
        , frameNum_(0)
        , calibrating_(false)
        , calibDone_(false)
        , trackingActive_(false)
        , templatesCollected_(0)
        , calibFrameCount_(0)
        , trackerConf_(-1.f)
    {
        // Load YOLO models
        yoloCalib_.load(params.yoloModelCalibration, params.yoloImgszCalibration,
                        params.yoloConfidence, params.yoloClassId,
                        params.backend, params.target);

        if (!params.yoloModelDetection.empty() &&
            params.yoloModelDetection == params.yoloModelCalibration &&
            params.yoloImgszDetection == params.yoloImgszCalibration)
        {
            // Same model — share (but YoloDetector copies Net, so just load again)
            yoloDetect_.load(params.yoloModelDetection, params.yoloImgszDetection,
                             params.yoloConfidence, params.yoloClassId,
                             params.backend, params.target);
        }
        else
        {
            yoloDetect_.load(params.yoloModelDetection, params.yoloImgszDetection,
                             params.yoloConfidence, params.yoloClassId,
                             params.backend, params.target);
        }

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
                    bank_.add(frame, detBbox);
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
                    printf("[BallTracker] Calibration frame %d: YOLO found bbox=(%d,%d,%dx%d) conf=%.4f but rejected: %s\n",
                           frameNum_, detBbox.x, detBbox.y, detBbox.width, detBbox.height, detConf, reason.c_str());
                }
            }
            else
            {
                printf("[BallTracker] Calibration frame %d: YOLO found nothing\n", frameNum_);
            }

            result.mode = BALL_TRACKER_MODE_CALIBRATING;

            // Check if calibration is done
            bool done = (templatesCollected_ >= params_.numTemplates) ||
                        (calibFrameCount_ >= params_.calibrationFrames);

            if (done && templatesCollected_ > 0)
            {
                tracker_->finalizeTemplates();
                tracker_->init(frame, lastBbox_);
                trackingActive_ = true;
                calibrating_ = false;
                calibDone_ = true;
            }
            else if (done && templatesCollected_ == 0)
            {
                // Calibration failed — stop calibrating, enter YOLO-only fallback
                printf("[BallTracker] Calibration FAILED: no ball detected in %d frames — falling back to YOLO-only\n",
                       calibFrameCount_);
                calibrating_ = false;
                calibDone_ = false;
                useTracker_ = false;
            }

            if (result.found)
            {
                result.normBbox = Rect2d(
                    (result.bbox.x + result.bbox.width * 0.5f) / frameW,
                    (result.bbox.y + result.bbox.height * 0.5f) / frameH,
                    (float)result.bbox.width / frameW,
                    (float)result.bbox.height / frameH
                );
            }
            return result;
        }

        // ==============================================================
        // Should YOLO run this frame?
        // ==============================================================
        bool runYolo = !trackingActive_
            || (params_.yoloPeriodic > 0 && frameNum_ % params_.yoloPeriodic == 0)
            || (params_.redetectInterval > 0 && frameNum_ % params_.redetectInterval == 0);

        if (!useTracker_)
            runYolo = true;

        // ==============================================================
        // Phase 1: TrackerNano update
        // ==============================================================
        if (trackingActive_ && tracker_ && useTracker_)
        {
            Rect trkBbox;
            bool success = tracker_->update(frame, trkBbox);
            trackerConf_ = tracker_->getTrackingScore();

            printf("[BallTracker] Frame %d: TrackerNano score=%.4f bbox=(%d,%d,%dx%d)\n",
                   frameNum_, trackerConf_, trkBbox.x, trkBbox.y, trkBbox.width, trkBbox.height);

            std::string reason;
            bool sane = isBboxSane(trkBbox, frameW, frameH,
                                   lastBbox_, hasLastBbox_,
                                   params_.maxBboxArea, params_.maxBboxJump, params_.maxAspect,
                                   reason);

            bool trkAccepted;
            if (params_.noTemplateValidation)
                trkAccepted = true;
            else if (sane)
                trkAccepted = bank_.validate(frame, trkBbox).first;
            else
                trkAccepted = false;

            if (success && trackerConf_ >= params_.confidenceThreshold && sane && trkAccepted)
            {
                lastBbox_ = trkBbox;
                hasLastBbox_ = true;
                result.bbox = trkBbox;
                result.found = true;
                result.mode = BALL_TRACKER_MODE_TRACKER;
                result.confidence = trackerConf_;
                // Don't override runYolo for calibrated trackers — allow periodic check
            }
            else
            {
                trackingActive_ = false;
                runYolo = true;
            }
        }

        // ==============================================================
        // Phase 2: YOLO detection (also runs on periodic frames for drift correction)
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
                    detBbox = Rect();
                }
            }

            if (detBbox.width > 0 && detBbox.height > 0)
            {
                // Appearance validation
                bool accepted = true;
                if (useTracker_ && !params_.noTemplateValidation)
                {
                    accepted = bank_.validate(frame, detBbox).first;
                }

                if (accepted)
                {
                    lastBbox_ = detBbox;
                    hasLastBbox_ = true;
                    result.bbox = detBbox;
                    result.found = true;
                    result.mode = BALL_TRACKER_MODE_YOLO;
                    result.confidence = detConf;

                    bank_.add(frame, detBbox);

                    // Re-initialise tracker
                    if (useTracker_ && calibDone_ && tracker_)
                    {
                        tracker_->init(frame, detBbox);
                        trackingActive_ = true;
                    }
                }
            }
        }

        // If still nothing found
        if (!result.found)
        {
            result.mode = BALL_TRACKER_MODE_LOST;
            trackingActive_ = false;
        }
        else
        {
            result.normBbox = Rect2d(
                (result.bbox.x + result.bbox.width * 0.5f) / frameW,
                (result.bbox.y + result.bbox.height * 0.5f) / frameH,
                (float)result.bbox.width / frameW,
                (float)result.bbox.height / frameH
            );
        }

        return result;
    }

    bool isCalibrating() const CV_OVERRIDE { return calibrating_; }
    int getFrameCount() const CV_OVERRIDE { return frameNum_; }
    int getTemplatesCollected() const CV_OVERRIDE { return templatesCollected_; }

private:
    BallTrackerParams params_;
    TemplateBank bank_;

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
