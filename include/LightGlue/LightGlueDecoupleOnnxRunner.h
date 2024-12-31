#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <opencv4/opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace Lightglue{
#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */
    
struct Configuation
{
    std::string lightglue_path;
    std::string extractor_path;

    std::string extractor_type;
    bool isgray = true;

    int image_size;
    float threshold;            // 匹配阈值0 0.5
    std::string device;
};

class LightGlueDecoupleOnnxRunner{
public:
    using Ptr = std::shared_ptr<LightGlueDecoupleOnnxRunner>;

    LightGlueDecoupleOnnxRunner(Configuation &config);

    // 初始化onnxruntime环境
    int init_ortenv();
    
    // 特征提取及匹配入口
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> 
        InferenceImage(const cv::Mat &srcImage, const cv::Mat &destImage);

private:
    cv::Mat PreProcess(cv::Mat &img, float &scale);    

    std::vector<Ort::Value> extractor_inference(cv::Mat &img);
    
    // 返回 特征点集合，描述子集合
    std::pair<std::vector<cv::Point2f>, float*> postprocess(std::vector<Ort::Value> &tensor, int w, int h);

    void match_inference(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1);

    void match_postprocess(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1);

private:
    Configuation cfg;
    float scale[2];

    Ort::Env env0, env1;
    Ort::SessionOptions session_options0, session_options1;
    std::unique_ptr<Ort::Session> extract_session, matcher_session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char *> ExtractorInputNodeNames;
    std::vector<std::vector<int64_t>> ExtractorInputNodeShapes;
    std::vector<char *> ExtractorOutputNodeNames;
    std::vector<std::vector<int64_t>> ExtractorOutputNodeShapes;

    std::vector<char *> MatcherInputNodeNames;
    std::vector<std::vector<int64_t>> MatcherInputNodeShapes;
    std::vector<char *> MatcherOutputNodeNames;
    std::vector<std::vector<int64_t>> MatcherOutputNodeShapes;

    std::vector<Ort::Value> extractor_outputtensors0, extractor_outputtensors1;
    std::vector<Ort::Value> matcher_outputtensors;

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;
};
}