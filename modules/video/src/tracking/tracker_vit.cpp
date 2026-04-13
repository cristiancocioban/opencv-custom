// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Author, PengyuLiu, 1872918507@qq.com

#include "../precomp.hpp"
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif

namespace cv {

TrackerVit::TrackerVit()
{
    // nothing
}

TrackerVit::~TrackerVit()
{
    // nothing
}

TrackerVit::Params::Params()
{
    net = "vitTracker.onnx";
    meanvalue = Scalar{0.485, 0.456, 0.406}; // normalized mean (already divided by 255)
    stdvalue = Scalar{0.229, 0.224, 0.225};  // normalized std (already divided by 255)
    maxTemplates = 0;  // 0 = unlimited
#ifdef HAVE_OPENCV_DNN
    backend = dnn::DNN_BACKEND_DEFAULT;
    target = dnn::DNN_TARGET_CPU;
#else
    backend = -1;  // invalid value
    target = -1;  // invalid value
#endif
    tracking_score_threshold = 0.20f; // safe threshold to filter out black frames
}

#ifdef HAVE_OPENCV_DNN

class TrackerVitImpl : public TrackerVit
{
public:
    TrackerVitImpl(const TrackerVit::Params& parameters)
        : params(parameters)
    {
        net = dnn::readNet(parameters.net);
        CV_Assert(!net.empty());

        net.setPreferableBackend(parameters.backend);
        net.setPreferableTarget(parameters.target);

        i2bp.mean = parameters.meanvalue * 255.0;
        i2bp.scalefactor = (1.0 / parameters.stdvalue) * (1 / 255.0);
        tracking_score_threshold = parameters.tracking_score_threshold;
    }

    TrackerVitImpl(const dnn::Net& model, Scalar meanvalue, Scalar stdvalue, float _tracking_score_threshold)
    {
        CV_Assert(!model.empty());

        net = model;
        i2bp.mean = meanvalue * 255.0;
        i2bp.scalefactor = (1.0 / stdvalue) * (1 / 255.0);
        tracking_score_threshold = _tracking_score_threshold;
    }

    void init(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    bool update(InputArray image, Rect& boundingBox) CV_OVERRIDE;
    float getTrackingScore() CV_OVERRIDE;

    void initFromEmbedding(const Rect& boundingBox, InputArray templateBlob, const Size& imageSize) CV_OVERRIDE;
    void addTemplate(InputArray image, const Rect& boundingBox) CV_OVERRIDE;
    void addTemplateEmbedding(InputArray templateBlob) CV_OVERRIDE;
    void finalizeTemplates() CV_OVERRIDE;
    Mat getTemplateEmbedding(int idx = 0) const CV_OVERRIDE;
    int getTemplateCount() const CV_OVERRIDE;

    Rect rect_last;
    float tracking_score;

    float tracking_score_threshold;
    dnn::Image2BlobParams i2bp;

protected:
    void preprocess(const Mat& src, Mat& dst, Size size);
    Mat encodeTemplate(const Mat& image, const Rect& boundingBox);

    const Size searchSize{256, 256};
    const Size templateSize{128, 128};

    TrackerVit::Params params;
    Mat hanningWindow;
    dnn::Net net;

    std::vector<Mat> templateBlobs;        // preprocessed template blobs
    bool calibrationFinalized = false;
    Rect frozenRect = {0, 0, 0, 0};       // frozen bbox size from calibration
};

static int crop_image(const Mat& src, Mat& dst, Rect box, int factor)
{
    int x = box.x, y = box.y, w = box.width, h = box.height;
    int crop_sz = cvCeil(sqrt(w * h) * factor);

    int x1 = x + (w - crop_sz) / 2;
    int x2 = x1 + crop_sz;
    int y1 = y + (h - crop_sz) / 2;
    int y2 = y1 + crop_sz;

    int x1_pad = std::max(0, -x1);
    int y1_pad = std::max(0, -y1);
    int x2_pad = std::max(x2 - src.size[1] + 1, 0);
    int y2_pad = std::max(y2 - src.size[0] + 1, 0);

    Rect roi(x1 + x1_pad, y1 + y1_pad, x2 - x2_pad - x1 - x1_pad, y2 - y2_pad - y1 - y1_pad);
    Mat im_crop = src(roi);
    copyMakeBorder(im_crop, dst, y1_pad, y2_pad, x1_pad, x2_pad, BORDER_CONSTANT);

    return crop_sz;
}

void TrackerVitImpl::preprocess(const Mat& src, Mat& dst, Size size)
{
    Mat img;
    resize(src, img, size);

    dst = dnn::blobFromImageWithParams(img, i2bp);
}

static Mat hann1d(int sz, bool centered = true) {
    Mat hanningWindow(sz, 1, CV_32FC1);
    float* data = hanningWindow.ptr<float>(0);

    if(centered) {
        for(int i = 0; i < sz; i++) {
            float val = 0.5f * (1.f - std::cos(static_cast<float>(2 * M_PI / (sz + 1)) * (i + 1)));
            data[i] = val;
        }
    }
    else {
        int half_sz = sz / 2;
        for(int i = 0; i <= half_sz; i++) {
            float val = 0.5f * (1.f + std::cos(static_cast<float>(2 * M_PI / (sz + 2)) * i));
            data[i] = val;
            data[sz - 1 - i] = val;
        }
    }

    return hanningWindow;
}

static Mat hann2d(Size size, bool centered = true) {
    int rows = size.height;
    int cols = size.width;

    Mat hanningWindowRows = hann1d(rows, centered);
    Mat hanningWindowCols = hann1d(cols, centered);

    Mat hanningWindow = hanningWindowRows * hanningWindowCols.t();

    return hanningWindow;
}

Mat TrackerVitImpl::encodeTemplate(const Mat& image, const Rect& boundingBox)
{
    Mat crop;
    crop_image(image, crop, boundingBox, 2);
    Mat blob;
    preprocess(crop, blob, templateSize);
    return blob.clone();
}

void TrackerVitImpl::init(InputArray image_, const Rect &boundingBox_)
{
    Mat image = image_.getMat();

    templateBlobs.clear();
    Mat blob = encodeTemplate(image, boundingBox_);
    templateBlobs.push_back(blob);
    net.setInput(blob, "template");

    hanningWindow = hann2d(Size(16, 16), true);
    rect_last = boundingBox_;
    frozenRect = boundingBox_;
    calibrationFinalized = true;

    fprintf(stderr, "[TrackerVit::init] 1 template stored, frozenRect=[%d x %d]\n",
        frozenRect.width, frozenRect.height);
}

void TrackerVitImpl::initFromEmbedding(const Rect& boundingBox, InputArray templateBlob, const Size& /*imageSize*/)
{
    bool hadTemplates = calibrationFinalized && !templateBlobs.empty();

    hanningWindow = hann2d(Size(16, 16), true);
    rect_last = boundingBox;
    frozenRect = boundingBox;

    if (!hadTemplates)
    {
        templateBlobs.clear();
        templateBlobs.push_back(templateBlob.getMat().clone());
    }

    net.setInput(templateBlobs[0], "template");
    calibrationFinalized = true;

    fprintf(stderr, "[TrackerVit::initFromEmbedding] %d template(s) active, frozenRect=[%d x %d], pos=[%d, %d]\n",
        (int)templateBlobs.size(), frozenRect.width, frozenRect.height, frozenRect.x, frozenRect.y);
}

void TrackerVitImpl::addTemplate(InputArray image_, const Rect& boundingBox)
{
    CV_Assert(!calibrationFinalized && "Cannot add templates after finalizeTemplates() or init()");
    if (params.maxTemplates > 0)
        CV_Assert((int)templateBlobs.size() < params.maxTemplates);

    Mat image = image_.getMat();
    Mat blob = encodeTemplate(image, boundingBox);
    templateBlobs.push_back(blob);
}

void TrackerVitImpl::addTemplateEmbedding(InputArray templateBlob)
{
    CV_Assert(!calibrationFinalized && "Cannot add templates after finalizeTemplates() or init()");
    if (params.maxTemplates > 0)
        CV_Assert((int)templateBlobs.size() < params.maxTemplates);

    templateBlobs.push_back(templateBlob.getMat().clone());
}

void TrackerVitImpl::finalizeTemplates()
{
    CV_Assert(!templateBlobs.empty() && "At least one template must be added before finalizing");
    calibrationFinalized = true;
    net.setInput(templateBlobs[0], "template");
    fprintf(stderr, "[TrackerVit::finalizeTemplates] %d template(s) LOCKED\n",
        (int)templateBlobs.size());
}

Mat TrackerVitImpl::getTemplateEmbedding(int idx) const
{
    CV_Assert(idx >= 0 && idx < (int)templateBlobs.size());
    return templateBlobs[idx].clone();
}

int TrackerVitImpl::getTemplateCount() const
{
    return (int)templateBlobs.size();
}

bool TrackerVitImpl::update(InputArray image_, Rect &boundingBoxRes)
{
    CV_Assert(calibrationFinalized && "Tracker must be initialized before calling update()");

    Mat image = image_.getMat();
    Mat crop;
    int crop_size = crop_image(image, crop, rect_last, 4);
    Mat searchBlob;
    preprocess(crop, searchBlob, searchSize);

    // Cross-correlate against ALL stored frozen templates, keep the best match
    float bestScore = -1.f;
    float bestCx = 0, bestCy = 0;
    size_t bestTemplateIdx = 0;

    std::vector<String> outputName = {"output1", "output2", "output3"};

    fprintf(stderr, "[TrackerVit::update] %d FROZEN template(s), frozenSz=[%d x %d], rect_last=[%d,%d,%dx%d]\n",
        (int)templateBlobs.size(), frozenRect.width, frozenRect.height,
        rect_last.x, rect_last.y, rect_last.width, rect_last.height);

    for (size_t t = 0; t < templateBlobs.size(); t++)
    {
        net.setInput(templateBlobs[t], "template");
        net.setInput(searchBlob, "search");
        std::vector<Mat> outs;
        net.forward(outs, outputName);
        CV_Assert(outs.size() == 3);

        Mat conf_map = outs[0].reshape(0, {16, 16});
        Mat size_map = outs[1].reshape(0, {2, 16, 16});
        Mat offset_map = outs[2].reshape(0, {2, 16, 16});

        multiply(conf_map, hanningWindow, conf_map);

        double maxVal;
        Point maxLoc;
        minMaxLoc(conf_map, nullptr, &maxVal, nullptr, &maxLoc);

        fprintf(stderr, "  template[%d] score=%.4f peakPos=[%d, %d]\n",
            (int)t, (float)maxVal, maxLoc.x, maxLoc.y);

        if ((float)maxVal > bestScore)
        {
            bestScore = (float)maxVal;
            bestTemplateIdx = t;

            if (bestScore >= tracking_score_threshold)
            {
                bestCx = (maxLoc.x + offset_map.at<float>(0, maxLoc.y, maxLoc.x)) / 16;
                bestCy = (maxLoc.y + offset_map.at<float>(1, maxLoc.y, maxLoc.x)) / 16;
            }
        }
    }

    fprintf(stderr, "  BEST template[%d] score=%.4f\n", (int)bestTemplateIdx, bestScore);

    tracking_score = bestScore;

    if (tracking_score >= tracking_score_threshold)
    {
        // Update position from the best match, but use FROZEN size
        int x0 = rect_last.x + (rect_last.width - crop_size) / 2;
        int y0 = rect_last.y + (rect_last.height - crop_size) / 2;

        float x1 = bestCx - (float)frozenRect.width / (2.0f * crop_size);
        float y1 = bestCy - (float)frozenRect.height / (2.0f * crop_size);

        rect_last.x = cvFloor(x1 * crop_size + x0);
        rect_last.y = cvFloor(y1 * crop_size + y0);
        rect_last.width = frozenRect.width;    // FROZEN — no size drift
        rect_last.height = frozenRect.height;   // FROZEN — no size drift

        fprintf(stderr, "  result pos=[%d, %d] size=[%d x %d] (FROZEN)\n",
            rect_last.x, rect_last.y, rect_last.width, rect_last.height);

        boundingBoxRes = rect_last;
        return true;
    }
    else
    {
        fprintf(stderr, "  BELOW THRESHOLD (%.4f < %.4f) — tracking lost\n",
            tracking_score, tracking_score_threshold);
        return false;
    }
}

float TrackerVitImpl::getTrackingScore()
{
    return tracking_score;
}

Ptr<TrackerVit> TrackerVit::create(const TrackerVit::Params& parameters)
{
    return makePtr<TrackerVitImpl>(parameters);
}

Ptr<TrackerVit> TrackerVit::create(const dnn::Net& model, Scalar meanvalue, Scalar stdvalue, float tracking_score_threshold)
{
    return makePtr<TrackerVitImpl>(model, meanvalue, stdvalue, tracking_score_threshold);
}

#else  // OPENCV_HAVE_DNN
Ptr<TrackerVit> TrackerVit::create(const TrackerVit::Params& parameters)
{
    CV_UNUSED(parameters);
    CV_Error(Error::StsNotImplemented, "to use vittrack, the tracking module needs to be built with opencv_dnn !");
}
#endif  // OPENCV_HAVE_DNN
}
