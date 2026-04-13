// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the https://github.com/HonglinChu/NanoTrack/blob/master/ncnn_macos_nanotrack/nanotrack.cpp
// Author, HongLinChu, 1628464345@qq.com
// Adapt to OpenCV, ZihaoMu: zihaomu@outlook.com

// Link to original inference code: https://github.com/HonglinChu/NanoTrack
// Link to original training repo: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack

#include "../precomp.hpp"
#include <deque>
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

TrackerNano::TrackerNano()
{
    // nothing
}

TrackerNano::~TrackerNano()
{
    // nothing
}

TrackerNano::Params::Params()
{
    backbone = "backbone.onnx";
    neckhead = "neckhead.onnx";
    maxTemplates = 0;  // 0 = unlimited
    searchCrops = 5;   // 1 = single center only, 5 = full multi-crop
    earlyExitScore = 0.85f;
    motionHistory = 5;
#ifdef HAVE_OPENCV_DNN
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    backend = -1;  // invalid value
    target = -1;  // invalid value
#endif
}

#ifdef HAVE_OPENCV_DNN
static void softmax(const Mat& src, Mat& dst)
{
    Mat maxVal;
    cv::max(src.row(1), src.row(0), maxVal);

    src.row(1) -= maxVal;
    src.row(0) -= maxVal;

    exp(src, dst);

    Mat sumVal = dst.row(0) + dst.row(1);
    dst.row(0) = dst.row(0) / sumVal;
    dst.row(1) = dst.row(1) / sumVal;
}

static float sizeCal(float w, float h)
{
    float pad = (w + h) * 0.5f;
    float sz2 = (w + pad) * (h + pad);
    return sqrt(sz2);
}

static Mat sizeCal(const Mat& w, const Mat& h)
{
    Mat pad = (w + h) * 0.5;
    Mat sz2 = (w + pad).mul((h + pad));

    cv::sqrt(sz2, sz2);
    return sz2;
}

// Similar python code: r = np.maximum(r, 1. / r) # r is matrix
static void elementReciprocalMax(Mat& srcDst)
{
    size_t totalV = srcDst.total();
    float* ptr = srcDst.ptr<float>(0);
    for (size_t i = 0; i < totalV; i++)
    {
        float val = *(ptr + i);
        *(ptr + i) = std::max(val, 1.0f/val);
    }
}

class TrackerNanoImpl : public TrackerNano
{
public:
    TrackerNanoImpl(const TrackerNano::Params& parameters)
        : params(parameters)
    {
        backbone = dnn::readNet(parameters.backbone);
        neckhead = dnn::readNet(parameters.neckhead);

        CV_Assert(!backbone.empty());
        CV_Assert(!neckhead.empty());

        backbone.setPreferableBackend(parameters.backend);
        backbone.setPreferableTarget(parameters.target);
        neckhead.setPreferableBackend(parameters.backend);
        neckhead.setPreferableTarget(parameters.target);
    }

    TrackerNanoImpl(const dnn::Net& _backbone, const dnn::Net& _neckhead)
    {
        CV_Assert(!_backbone.empty());
        CV_Assert(!_neckhead.empty());

        backbone = _backbone;
        neckhead = _neckhead;
    }

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    float getTrackingScore() CV_OVERRIDE;

    void initFromEmbedding(const Rect& boundingBox, InputArray embedding, const Size& imageSize) CV_OVERRIDE;
    void addTemplate(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    void addTemplateEmbedding(InputArray embedding) CV_OVERRIDE;
    void finalizeTemplates() CV_OVERRIDE;
    Mat getTemplateEmbedding(int idx = 0) const CV_OVERRIDE;
    int getTemplateCount() const CV_OVERRIDE;

    // Save the target bounding box for each frame.
    std::vector<float> targetSz = {0, 0};  // H and W of bounding box
    std::vector<float> targetPos = {0, 0}; // center point of bounding box (x, y)
    float tracking_score;

    struct trackerConfig
    {
        float windowInfluence = 0.455f;
        float lr = 0.37f;
        float contextAmount = 0.5;
        bool swapRB = true;
        int totalStride = 16;
        float penaltyK = 0.055f;
    };

protected:
    const int exemplarSize = 127;
    const int instanceSize = 255;

    TrackerNano::Params params;
    trackerConfig trackState;
    int scoreSize;
    bool scoreSizeProbed = false;

    void probeScoreSize();
    Size imgSize = {0, 0};
    Mat hanningWindow;
    Mat grid2searchX, grid2searchY;

    dnn::Net backbone, neckhead;
    Mat image;

    std::vector<Mat> templateEmbeddings;
    bool calibrationFinalized = false;
    std::vector<float> frozenTargetSz = {0, 0}; // locked at finalization, never mutated by update()
    bool frozenSzInitialized = false;            // true after frozenTargetSz is set for the first time

    // Motion history for velocity-based prediction
    std::deque<Point2f> posHistory;

    // Run backbone+neckhead at a specific search center, return best raw cls score
    // and populate outPos with the result position in image coords if successful.
    struct CropResult {
        float rawClsScore;
        float bestPscore;
        float resX, resY;
        size_t templateIdx;
    };
    CropResult evaluateCrop(const Point2f& searchCenter, float scale_z, float sx,
                            float scaledW, float scaledH);

    Mat encodeTemplate(Mat& img, const Rect& boundingBox);
    void initState(const Rect& boundingBox, const Size& imageSize);
    void getSubwindow(Mat& dstCrop, Mat& srcImg, int originalSz, int resizeSz);
    void getSubwindowAt(Mat& dstCrop, Mat& srcImg, const Point2f& center, int originalSz, int resizeSz);
    void generateGrids();
};

void TrackerNanoImpl::generateGrids()
{
    int sz = scoreSize;
    const int sz2 = sz / 2;

    std::vector<float> x1Vec(sz, 0);

    for (int i = 0; i < sz; i++)
    {
        x1Vec[i] = (float)(i - sz2);
    }

    Mat x1M(1, sz, CV_32FC1, x1Vec.data());

    cv::repeat(x1M, sz, 1, grid2searchX);
    cv::repeat(x1M.t(), 1, sz, grid2searchY);

    grid2searchX *= trackState.totalStride;
    grid2searchY *= trackState.totalStride;

    grid2searchX += instanceSize/2;
    grid2searchY += instanceSize/2;
}

void TrackerNanoImpl::probeScoreSize()
{
    if (scoreSizeProbed)
        return;

    // Run the actual backbone on dummy images to get correct feature shapes,
    // then run neckhead to detect the output score map size.
    // This handles any NanoTrack version (V2=16×16, V3=15×15, etc.)
    Mat dummyTemplateImg = Mat::zeros(exemplarSize, exemplarSize, CV_8UC3);
    Mat dummySearchImg = Mat::zeros(instanceSize, instanceSize, CV_8UC3);

    Mat blobT = dnn::blobFromImage(dummyTemplateImg, 1.0, Size(), Scalar(), trackState.swapRB);
    backbone.setInput(blobT);
    Mat tmplFeat = backbone.forward();

    Mat blobS = dnn::blobFromImage(dummySearchImg, 1.0, Size(), Scalar(), trackState.swapRB);
    backbone.setInput(blobS);
    Mat searchFeat = backbone.forward();

    neckhead.setInput(tmplFeat, "input1");
    neckhead.setInput(searchFeat, "input2");
    std::vector<String> outNames = {"output1"};
    std::vector<Mat> outs;
    neckhead.forward(outs, outNames);

    // output1 shape is [1, 2, scoreSize, scoreSize]
    int totalElements = (int)outs[0].total();
    scoreSize = (int)std::sqrt(totalElements / 2);  // 2 channels
    scoreSizeProbed = true;

    fprintf(stderr, "[TrackerNano::probeScoreSize] Detected scoreSize=%d from neckhead output (%d elements)\n",
        scoreSize, totalElements);
}

void TrackerNanoImpl::initState(const Rect& boundingBox, const Size& imageSize)
{
    probeScoreSize();
    trackState = trackerConfig();

    // convert Rect from left-top to center.
    targetPos[0] = float(boundingBox.x) + float(boundingBox.width) * 0.5f;
    targetPos[1] = float(boundingBox.y) + float(boundingBox.height) * 0.5f;

    targetSz[0] = float(boundingBox.width);
    targetSz[1] = float(boundingBox.height);

    imgSize = imageSize;

    createHanningWindow(hanningWindow, Size(scoreSize, scoreSize), CV_32F);
    generateGrids();
}

Mat TrackerNanoImpl::encodeTemplate(Mat& img, const Rect& boundingBox)
{
    float cx = float(boundingBox.x) + float(boundingBox.width) * 0.5f;
    float cy = float(boundingBox.y) + float(boundingBox.height) * 0.5f;
    float w = float(boundingBox.width);
    float h = float(boundingBox.height);

    float sumSz = w + h;
    float wExtent = w + trackState.contextAmount * sumSz;
    float hExtent = h + trackState.contextAmount * sumSz;
    int sz = int(cv::sqrt(wExtent * hExtent));

    // Temporarily set targetPos for getSubwindow, then restore
    auto savedPos = targetPos;
    auto savedSz = targetSz;
    targetPos = {cx, cy};
    targetSz = {w, h};

    Mat crop;
    getSubwindow(crop, img, sz, exemplarSize);

    targetPos = savedPos;
    targetSz = savedSz;

    Mat blob = dnn::blobFromImage(crop, 1.0, Size(), Scalar(), trackState.swapRB);
    backbone.setInput(blob);
    return backbone.forward().clone();
}

void TrackerNanoImpl::init(InputArray image_, const Rect &boundingBox_)
{
    image = image_.getMat().clone();

    initState(boundingBox_, image.size());

    templateEmbeddings.clear();
    Mat emb = encodeTemplate(image, boundingBox_);
    templateEmbeddings.push_back(emb);
    neckhead.setInput(emb, "input1");
    calibrationFinalized = true;
    frozenTargetSz = targetSz;
    frozenSzInitialized = true;
    posHistory.clear();
    posHistory.push_back(Point2f(targetPos[0], targetPos[1]));
    fprintf(stderr, "[TrackerNano::init] 1 template stored, frozenTargetSz=[%.1f x %.1f]\n",
        frozenTargetSz[0], frozenTargetSz[1]);
}

void TrackerNanoImpl::initFromEmbedding(const Rect& boundingBox, InputArray embedding, const Size& imageSize)
{
    // If templates were already finalized (recovery path), preserve them.
    // Only reset position/size state, not the template bank.
    bool hadTemplates = calibrationFinalized && !templateEmbeddings.empty();

    initState(boundingBox, imageSize);

    if (!hadTemplates)
    {
        // Fresh init with a single embedding
        templateEmbeddings.clear();
        Mat emb = embedding.getMat().clone();
        templateEmbeddings.push_back(emb);
    }
    // else: keep existing templateEmbeddings from calibration intact

    neckhead.setInput(templateEmbeddings[0], "input1");
    calibrationFinalized = true;

    // Always update frozen size from the YOLO bbox — it's more accurate
    // than keeping a stale calibration size as the ball changes apparent size.
    frozenTargetSz = targetSz;
    frozenSzInitialized = true;

    posHistory.clear();
    posHistory.push_back(Point2f(targetPos[0], targetPos[1]));
    fprintf(stderr, "[TrackerNano::initFromEmbedding] %d template(s), recovery=%s, frozenTargetSz=[%.1f x %.1f], pos=[%.1f, %.1f]\n",
        (int)templateEmbeddings.size(), hadTemplates ? "YES" : "NO",
        frozenTargetSz[0], frozenTargetSz[1], targetPos[0], targetPos[1]);
}

void TrackerNanoImpl::addTemplate(InputArray image_, const Rect& boundingBox)
{
    CV_Assert(!calibrationFinalized && "Cannot add templates after finalizeTemplates() or init()");
    if (params.maxTemplates > 0)
        CV_Assert((int)templateEmbeddings.size() < params.maxTemplates);

    // Ensure tracker config is initialized for encoding
    if (templateEmbeddings.empty())
        trackState = trackerConfig();

    Mat img = image_.getMat().clone();
    Mat emb = encodeTemplate(img, boundingBox);
    templateEmbeddings.push_back(emb);
}

void TrackerNanoImpl::addTemplateEmbedding(InputArray embedding)
{
    CV_Assert(!calibrationFinalized && "Cannot add templates after finalizeTemplates() or init()");
    if (params.maxTemplates > 0)
        CV_Assert((int)templateEmbeddings.size() < params.maxTemplates);

    templateEmbeddings.push_back(embedding.getMat().clone());
}

void TrackerNanoImpl::finalizeTemplates()
{
    CV_Assert(!templateEmbeddings.empty() && "At least one template must be added before finalizing");
    calibrationFinalized = true;
    neckhead.setInput(templateEmbeddings[0], "input1");
    fprintf(stderr, "[TrackerNano::finalizeTemplates] %d template(s) LOCKED\n",
        (int)templateEmbeddings.size());
}

Mat TrackerNanoImpl::getTemplateEmbedding(int idx) const
{
    CV_Assert(idx >= 0 && idx < (int)templateEmbeddings.size());
    return templateEmbeddings[idx].clone();
}

int TrackerNanoImpl::getTemplateCount() const
{
    return (int)templateEmbeddings.size();
}

void TrackerNanoImpl::getSubwindow(Mat& dstCrop, Mat& srcImg, int originalSz, int resizeSz)
{
    Scalar avgChans = mean(srcImg);
    Size imgSz = srcImg.size();
    int c = (originalSz + 1) / 2;

    int context_xmin = (int)(targetPos[0]) - c;
    int context_xmax = context_xmin + originalSz - 1;
    int context_ymin = (int)(targetPos[1]) - c;
    int context_ymax = context_ymin + originalSz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - imgSz.width + 1);
    int bottom_pad = std::max(0, context_ymax - imgSz.height + 1);

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    Mat cropImg;
    if (left_pad == 0 && top_pad == 0 && right_pad == 0 && bottom_pad == 0)
    {
        // Crop image without padding.
        cropImg = srcImg(cv::Rect(context_xmin, context_ymin,
                                  context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else // Crop image with padding, and the padding value is avgChans
    {
        cv::Mat tmpMat;
        cv::copyMakeBorder(srcImg, tmpMat, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, avgChans);
        cropImg = tmpMat(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    resize(cropImg, dstCrop, Size(resizeSz, resizeSz));
}

void TrackerNanoImpl::getSubwindowAt(Mat& dstCrop, Mat& srcImg, const Point2f& center, int originalSz, int resizeSz)
{
    Scalar avgChans = mean(srcImg);
    Size imgSz = srcImg.size();
    int c = (originalSz + 1) / 2;

    int context_xmin = (int)(center.x) - c;
    int context_xmax = context_xmin + originalSz - 1;
    int context_ymin = (int)(center.y) - c;
    int context_ymax = context_ymin + originalSz - 1;

    int left_pad = std::max(0, -context_xmin);
    int top_pad = std::max(0, -context_ymin);
    int right_pad = std::max(0, context_xmax - imgSz.width + 1);
    int bottom_pad = std::max(0, context_ymax - imgSz.height + 1);

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    Mat cropImg;
    if (left_pad == 0 && top_pad == 0 && right_pad == 0 && bottom_pad == 0)
    {
        cropImg = srcImg(cv::Rect(context_xmin, context_ymin,
                                  context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
    {
        cv::Mat tmpMat;
        cv::copyMakeBorder(srcImg, tmpMat, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, avgChans);
        cropImg = tmpMat(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    resize(cropImg, dstCrop, Size(resizeSz, resizeSz));
}

TrackerNanoImpl::CropResult TrackerNanoImpl::evaluateCrop(
    const Point2f& searchCenter, float scale_z, float sx,
    float scaledW, float scaledH)
{
    Mat crop;
    getSubwindowAt(crop, image, searchCenter, int(sx), instanceSize);

    Mat blob = dnn::blobFromImage(crop, 1.0, Size(), Scalar(), trackState.swapRB);
    backbone.setInput(blob);
    Mat xf = backbone.forward().clone();

    CropResult best;
    best.rawClsScore = 0.f;
    best.bestPscore = -1.f;
    best.resX = searchCenter.x;
    best.resY = searchCenter.y;
    best.templateIdx = 0;

    std::vector<String> outputName = {"output1", "output2"};

    for (size_t t = 0; t < templateEmbeddings.size(); t++)
    {
        neckhead.setInput(templateEmbeddings[t], "input1");
        neckhead.setInput(xf, "input2");
        std::vector<Mat> outs;
        neckhead.forward(outs, outputName);
        CV_Assert(outs.size() == 2);

        Mat clsScore = outs[0].reshape(0, {2, scoreSize, scoreSize});
        Mat bboxPred = outs[1].reshape(0, {4, scoreSize, scoreSize});

        Mat scoreSoftmax;
        softmax(clsScore, scoreSoftmax);

        Mat score = scoreSoftmax.row(1).reshape(0, {scoreSize, scoreSize});

        Mat pX1 = grid2searchX - bboxPred.row(0).reshape(0, {scoreSize, scoreSize});
        Mat pY1 = grid2searchY - bboxPred.row(1).reshape(0, {scoreSize, scoreSize});
        Mat pX2 = grid2searchX + bboxPred.row(2).reshape(0, {scoreSize, scoreSize});
        Mat pY2 = grid2searchY + bboxPred.row(3).reshape(0, {scoreSize, scoreSize});

        Mat sc = sizeCal(pX2 - pX1, pY2 - pY1) / sizeCal(scaledW, scaledH);
        elementReciprocalMax(sc);

        float ratioVal = scaledW / scaledH;
        Mat ratioM(scoreSize, scoreSize, CV_32FC1, Scalar::all(ratioVal));
        Mat rc = ratioM / ((pX2 - pX1) / (pY2 - pY1));
        elementReciprocalMax(rc);

        Mat penalty;
        exp(((rc.mul(sc) - 1) * trackState.penaltyK * (-1)), penalty);
        Mat pscore = penalty.mul(score);

        pscore = pscore * (1.0 - trackState.windowInfluence) + hanningWindow * trackState.windowInfluence;

        double maxVal;
        int maxID[2] = {0, 0};
        minMaxIdx(pscore, 0, &maxVal, 0, maxID);

        float rawCls = score.at<float>(maxID);

        if ((float)maxVal > best.bestPscore)
        {
            best.bestPscore = (float)maxVal;
            best.rawClsScore = rawCls;
            best.templateIdx = t;

            // Convert peak position back to image coordinates
            float x1Val = pX1.at<float>(maxID);
            float x2Val = pX2.at<float>(maxID);
            float y1Val = pY1.at<float>(maxID);
            float y2Val = pY2.at<float>(maxID);

            float predXs = (x1Val + x2Val) / 2;
            float predYs = (y1Val + y2Val) / 2;

            float diffXs = (predXs - instanceSize / 2) / scale_z;
            float diffYs = (predYs - instanceSize / 2) / scale_z;

            best.resX = searchCenter.x + diffXs;
            best.resY = searchCenter.y + diffYs;
        }
    }
    return best;
}

bool TrackerNanoImpl::update(InputArray image_, Rect &boundingBoxRes)
{
    CV_Assert(calibrationFinalized && "Tracker must be initialized before calling update()");

    image = image_.getMat().clone();

    // Frozen target size for search region and penalties
    float tW = frozenTargetSz[0];
    float tH = frozenTargetSz[1];
    int targetSzSum = (int)(tW + tH);

    float wc = tW + trackState.contextAmount * targetSzSum;
    float hc = tH + trackState.contextAmount * targetSzSum;
    float sz = cv::sqrt(wc * hc);
    float scale_z = exemplarSize / sz;
    float sx = sz * (instanceSize / exemplarSize);
    float scaledW = tW * scale_z;
    float scaledH = tH * scale_z;

    // --- Compute motion-predicted position ---
    Point2f currentPos(targetPos[0], targetPos[1]);
    Point2f velocity(0, 0);
    if ((int)posHistory.size() >= 2)
    {
        // Average velocity over available history (up to 3-frame span)
        int span = std::min((int)posHistory.size() - 1, 3);
        Point2f oldest = posHistory[posHistory.size() - 1 - span];
        Point2f newest = posHistory[posHistory.size() - 1];
        velocity = (newest - oldest) * (1.0f / span);
    }
    Point2f predictedPos = currentPos + velocity;
    // Clamp to image bounds
    predictedPos.x = std::max(0.f, std::min((float)imgSize.width, predictedPos.x));
    predictedPos.y = std::max(0.f, std::min((float)imgSize.height, predictedPos.y));

    // --- Build search crop centers in priority order ---
    int maxCrops = params.searchCrops;
    float shiftAmount = sx * 1.0f;  // no overlap — tiles extend full search region size

    // Normalize velocity direction for perpendicular computation
    float vlen = cv::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
    Point2f vdir(0, 0), vperp(0, 0);
    if (vlen > 1.0f)
    {
        vdir = velocity * (1.0f / vlen);
        vperp = Point2f(-vdir.y, vdir.x);  // perpendicular (90 degrees)
    }

    std::vector<std::pair<Point2f, const char*>> cropCenters;

    // All crops are aligned along the velocity direction (beam pattern).
    // Crop 1: center (last known position) — always present
    // Crop 2+: stacked ahead in the direction of motion, no overlap
    //
    // With N crops and no overlap, reach = N * sx pixels along velocity.
    // For a 86px ball: 5 crops × 173px = 865px reach — covers a full dribble arc.

    static const char* cropNames[] = {"center", "beam_1", "beam_2", "beam_3", "beam_4"};

    cropCenters.push_back({currentPos, cropNames[0]});

    if (vlen > 1.0f)
    {
        for (int i = 1; i < maxCrops; i++)
        {
            Point2f beamCenter = currentPos + vdir * (shiftAmount * i);
            cropCenters.push_back({beamCenter, cropNames[std::min(i, 4)]});
        }
    }

    // Clamp all crop centers to image bounds
    for (auto& cc : cropCenters)
    {
        cc.first.x = std::max(0.f, std::min((float)imgSize.width, cc.first.x));
        cc.first.y = std::max(0.f, std::min((float)imgSize.height, cc.first.y));
    }

    fprintf(stderr, "[TrackerNano::update] %d template(s), frozenSz=[%.1f x %.1f], pos=[%.1f,%.1f], vel=[%.1f,%.1f], %d crop candidates\n",
        (int)templateEmbeddings.size(), tW, tH, currentPos.x, currentPos.y,
        velocity.x, velocity.y, (int)cropCenters.size());

    // --- Evaluate crops with early exit ---
    CropResult bestResult;
    bestResult.rawClsScore = 0.f;
    bestResult.bestPscore = -1.f;
    bestResult.resX = currentPos.x;
    bestResult.resY = currentPos.y;
    bestResult.templateIdx = 0;
    const char* bestCropName = "none";

    for (size_t c = 0; c < cropCenters.size(); c++)
    {
        const Point2f& center = cropCenters[c].first;
        const char* name = cropCenters[c].second;

        CropResult cr = evaluateCrop(center, scale_z, sx, scaledW, scaledH);

        fprintf(stderr, "  crop[%d](%s) center=[%.0f,%.0f] rawCls=%.4f tmpl=%d\n",
            (int)c, name, center.x, center.y, cr.rawClsScore, (int)cr.templateIdx);

        if (cr.rawClsScore > bestResult.rawClsScore)
        {
            bestResult = cr;
            bestCropName = name;
        }
    }

    fprintf(stderr, "  BEST crop=%s rawCls=%.4f tmpl=%d pos=[%.1f,%.1f]\n",
        bestCropName, bestResult.rawClsScore, (int)bestResult.templateIdx,
        bestResult.resX, bestResult.resY);

    tracking_score = bestResult.rawClsScore;

    // Update position — size stays frozen
    float resX = std::max(0.f, std::min((float)imgSize.width, bestResult.resX));
    float resY = std::max(0.f, std::min((float)imgSize.height, bestResult.resY));

    targetPos[0] = resX;
    targetPos[1] = resY;

    // Update motion history
    posHistory.push_back(Point2f(resX, resY));
    while ((int)posHistory.size() > params.motionHistory)
        posHistory.pop_front();

    // Output bbox uses frozen size
    float resW = std::max(10.f, std::min((float)imgSize.width, tW));
    float resH = std::max(10.f, std::min((float)imgSize.height, tH));

    boundingBoxRes = { int(resX - resW/2), int(resY - resH/2), int(resW), int(resH)};
    return true;
}

float TrackerNanoImpl::getTrackingScore()
{
    return tracking_score;
}

Ptr<TrackerNano> TrackerNano::create(const TrackerNano::Params& parameters)
{
    return makePtr<TrackerNanoImpl>(parameters);
}

Ptr<TrackerNano> TrackerNano::create(const dnn::Net& backbone, const dnn::Net& neckhead)
{
    return makePtr<TrackerNanoImpl>(backbone, neckhead);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerNano> TrackerNano::create(const TrackerNano::Params& parameters)
{
    CV_UNUSED(parameters);
    CV_Error(cv::Error::StsNotImplemented, "to use NanoTrack, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN
}
